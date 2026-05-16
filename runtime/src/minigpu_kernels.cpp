#include "minigpu_kernels.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef MINIGPU_DEFAULT_KERNEL_DIR
#define MINIGPU_DEFAULT_KERNEL_DIR "runtime/kernels"
#endif

namespace minigpu::kernels {

namespace {

/* Parse one 32-bit hex instruction word. */
std::uint32_t parse_hex_word(const std::string &value) {
    std::uint32_t word = 0;
    std::stringstream stream(value);
    stream >> std::hex >> word;
    if (!stream || !stream.eof()) {
        throw Error(Status::BadArgument);
    }
    return word;
}

/* Convert a YAML scalar into a uint32_t, accepting decimal or quoted 0x text. */
std::uint32_t yaml_u32(const YAML::Node &node, std::uint32_t fallback) {
    if (!node) {
        return fallback;
    }

    if (node.IsScalar()) {
        return static_cast<std::uint32_t>(std::stoul(node.as<std::string>(), nullptr, 0));
    }

    throw Error(Status::BadArgument);
}

/* Convert a YAML scalar into an active mask, with "auto" meaning all block lanes. */
std::uint32_t yaml_active_mask(const YAML::Node &node, std::uint32_t block_dim) {
    if (!node) {
        return block_dim >= 32u ? 0xffffffffu : ((1u << block_dim) - 1u);
    }

    const std::string text = node.as<std::string>();
    if (text == "auto") {
        return block_dim >= 32u ? 0xffffffffu : ((1u << block_dim) - 1u);
    }
    return static_cast<std::uint32_t>(std::stoul(text, nullptr, 0));
}

/* Return true when a path is already absolute. */
bool is_absolute_path(std::string_view path) {
    return !path.empty() && path.front() == '/';
}

/* Return the parent folder of a path. */
std::string parent_dir(std::string_view path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string_view::npos) {
        return ".";
    }
    if (slash == 0) {
        return "/";
    }
    return std::string(path.substr(0, slash));
}

/* Resolve a manifest file entry relative to the manifest itself. */
std::string resolve_artifact_path(std::string_view manifest_path, const std::string &file) {
    if (is_absolute_path(file)) {
        return file;
    }
    return parent_dir(manifest_path) + "/" + file;
}

/* Load a whole file as bytes. */
std::vector<std::uint8_t> read_binary_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Error(Status::NotFound);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        throw Error(Status::BadArgument);
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!file) {
        throw Error(Status::Transport);
    }
    return bytes;
}

/* Load raw big-endian instruction words as host-order instruction words. */
std::vector<std::uint8_t> read_binary_words_file(const std::string &path) {
    const std::vector<std::uint8_t> raw = read_binary_file(path);
    if ((raw.size() % sizeof(std::uint32_t)) != 0u) {
        throw Error(Status::BadArgument);
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(raw.size());
    for (std::size_t offset = 0; offset < raw.size(); offset += sizeof(std::uint32_t)) {
        const std::uint32_t word =
            (static_cast<std::uint32_t>(raw[offset]) << 24) |
            (static_cast<std::uint32_t>(raw[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(raw[offset + 2]) << 8) |
            static_cast<std::uint32_t>(raw[offset + 3]);
        const auto *word_bytes = reinterpret_cast<const std::uint8_t *>(&word);
        bytes.insert(bytes.end(), word_bytes, word_bytes + sizeof(word));
    }

    return bytes;
}

/* Load newline-separated 32-bit hex words as host-order instruction words. */
std::vector<std::uint8_t> read_hex_words_file(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw Error(Status::NotFound);
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (file >> token) {
        if (!token.empty() && token.front() == '#') {
            std::string ignored;
            std::getline(file, ignored);
            continue;
        }

        std::uint32_t word = parse_hex_word(token);
        const auto *word_bytes = reinterpret_cast<const std::uint8_t *>(&word);
        bytes.insert(bytes.end(), word_bytes, word_bytes + sizeof(word));
    }

    if (bytes.empty()) {
        throw Error(Status::BadArgument);
    }
    return bytes;
}

/* Load either raw .bin bytes or a text .hex file of 32-bit words. */
std::vector<std::uint8_t> read_kernel_file(const std::string &path) {
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".hex") == 0) {
        return read_hex_words_file(path);
    }
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".bin") == 0) {
        return read_binary_words_file(path);
    }
    return read_binary_file(path);
}

/* Load a compiler-generated kernel symbol map. */
std::unordered_map<std::string, DeviceAddress> read_symbol_map(const std::string &path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception &) {
        throw Error(Status::NotFound);
    }

    const YAML::Node symbols = root["symbols"];
    if (!symbols || !symbols.IsSequence()) {
        throw Error(Status::BadArgument);
    }

    std::unordered_map<std::string, DeviceAddress> out;
    for (const YAML::Node &entry : symbols) {
        out.emplace(entry["name"].as<std::string>(), static_cast<DeviceAddress>(yaml_u32(entry["pc"], 0)));
    }
    return out;
}

/* Return a file stem with a new extension. */
std::string replace_extension(const std::string &path, std::string_view extension) {
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + std::string(extension);
    }
    return path.substr(0, dot) + std::string(extension);
}

/* Parse one kernel entry inside an artifact block. */
PrecompiledKernel parse_kernel_entry(
    const YAML::Node &entry,
    const std::string &artifact_path,
    const std::vector<std::uint8_t> &program_bytes,
    const std::unordered_map<std::string, DeviceAddress> &symbols,
    const LaunchConfig &defaults) {
    PrecompiledKernel kernel;
    kernel.name = entry["name"].as<std::string>();
    kernel.program_id = artifact_path;
    kernel.op = entry["op"].as<std::string>();
    kernel.dtype = entry["dtype"].as<std::string>();
    const std::string entry_name =
        entry["entry"] ? entry["entry"].as<std::string>() : kernel.name;

    const auto symbol = symbols.find(entry_name);
    if (symbol == symbols.end()) {
        throw Error(Status::NotFound);
    }

    kernel.program_bytes = program_bytes;
    kernel.entry_pc = symbol->second;
    kernel.base_pc = symbol->second;
    kernel.default_grid_dim = yaml_u32(entry["grid_dim"], defaults.grid_dim);
    kernel.default_block_dim = yaml_u32(entry["block_dim"], defaults.block_dim);
    kernel.default_active_mask =
        yaml_active_mask(entry["active_mask"], kernel.default_block_dim);
    kernel.default_timeout_ms = yaml_u32(entry["timeout_ms"], defaults.timeout_ms);

    if (kernel.name.empty() || kernel.op.empty() ||
        kernel.dtype.empty() || artifact_path.empty() || kernel.program_bytes.empty()) {
        throw Error(Status::BadArgument);
    }
    return kernel;
}

/* Parse the runtime kernel manifest. */
std::vector<PrecompiledKernel> parse_manifest(std::string_view manifest_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(std::string(manifest_path));
    } catch (const YAML::Exception &) {
        throw Error(Status::NotFound);
    }

    const YAML::Node artifact_nodes = root["artifacts"];
    if (!artifact_nodes || !artifact_nodes.IsSequence()) {
        throw Error(Status::BadArgument);
    }

    LaunchConfig defaults;
    if (const YAML::Node defaults_node = root["defaults"]) {
        defaults.grid_dim = yaml_u32(defaults_node["grid_dim"], defaults.grid_dim);
        defaults.block_dim = yaml_u32(defaults_node["block_dim"], defaults.block_dim);
        defaults.active_mask =
            yaml_active_mask(defaults_node["active_mask"], defaults.block_dim);
        defaults.timeout_ms = yaml_u32(defaults_node["timeout_ms"], defaults.timeout_ms);
    } else {
        defaults.active_mask = yaml_active_mask(YAML::Node{}, defaults.block_dim);
    }

    std::vector<PrecompiledKernel> kernels;
    for (const YAML::Node &artifact : artifact_nodes) {
        if (!artifact.IsMap()) {
            throw Error(Status::BadArgument);
        }

        const std::string file = artifact["file"].as<std::string>();
        const std::string artifact_path = resolve_artifact_path(manifest_path, file);
        const std::string map_file = artifact["map"]
            ? artifact["map"].as<std::string>()
            : replace_extension(file, ".map");
        const std::string map_path = resolve_artifact_path(manifest_path, map_file);
        const std::vector<std::uint8_t> program_bytes = read_kernel_file(artifact_path);
        const auto symbols = read_symbol_map(map_path);

        const YAML::Node kernel_nodes = artifact["kernels"];
        if (!kernel_nodes || !kernel_nodes.IsSequence()) {
            throw Error(Status::BadArgument);
        }

        for (const YAML::Node &entry : kernel_nodes) {
            kernels.push_back(parse_kernel_entry(
                entry, artifact_path, program_bytes, symbols, defaults));
        }
    }

    return kernels;
}

} // namespace

/* Return the kernel artifact folder, allowing an environment override. */
std::string default_kernel_dir() {
    const char *override_dir = std::getenv("MINIGPU_KERNEL_DIR");
    if (override_dir && override_dir[0] != '\0') {
        return override_dir;
    }

    return MINIGPU_DEFAULT_KERNEL_DIR;
}

/* Return the default manifest path. */
std::string default_manifest_path() {
    const char *override_manifest = std::getenv("MINIGPU_KERNEL_MANIFEST");
    if (override_manifest && override_manifest[0] != '\0') {
        return override_manifest;
    }

    return default_kernel_dir() + "/kernels.yaml";
}

/* Register every kernel listed in a YAML manifest. */
void register_manifest(Context &context, std::string_view manifest_path) {
    for (auto &kernel : parse_manifest(manifest_path)) {
        context.register_kernel(std::move(kernel));
    }
}

/* Register all kernels listed in the default runtime kernel manifest. */
void register_builtin_kernels(Context &context) {
    register_manifest(context, default_manifest_path());
}

/* Resolve and launch a binary elementwise kernel using tensor addresses as args. */
void launch_elementwise_binary(
    Context &context,
    std::string_view op,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    const LaunchConfig *launch_config) {
    if (a.dtype != b.dtype || a.dtype != out.dtype ||
        a.elements != b.elements || a.elements != out.elements) {
        throw Error(Status::BadArgument);
    }
    if (a.elements > UINT32_MAX) {
        throw Error(Status::OutOfRange);
    }

    context.launch_kernel(
        std::string(op) + "." + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(b.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(static_cast<std::uint32_t>(a.elements)),
        },
        launch_config);
}

/* Resolve and launch a unary elementwise kernel using tensor addresses as args. */
void launch_elementwise_unary(
    Context &context,
    std::string_view op,
    const TensorView &a,
    const TensorView &out,
    const LaunchConfig *launch_config) {
    if (a.dtype != out.dtype || a.elements != out.elements) {
        throw Error(Status::BadArgument);
    }
    if (a.elements > UINT32_MAX) {
        throw Error(Status::OutOfRange);
    }

    context.launch_kernel(
        std::string(op) + "." + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(static_cast<std::uint32_t>(a.elements)),
        },
        launch_config);
}

/* Resolve and launch a matrix-multiply kernel with row-major tensor arguments. */
void launch_matmul(
    Context &context,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const LaunchConfig *launch_config) {
    if (a.dtype != b.dtype || a.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("matmul.") + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(b.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(m),
            KernelArg::u32(n),
            KernelArg::u32(k),
        },
        launch_config);
}

/* Resolve and launch a matrix-vector multiply kernel. */
void launch_gemv(
    Context &context,
    const TensorView &a,
    const TensorView &x,
    const TensorView &out,
    std::uint32_t m,
    std::uint32_t n,
    const LaunchConfig *launch_config) {
    if (a.dtype != x.dtype || a.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("gemv.") + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(x.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(m),
            KernelArg::u32(n),
        },
        launch_config);
}

/* Resolve and launch a dot product kernel. */
void launch_dot(
    Context &context,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    std::uint32_t n,
    const LaunchConfig *launch_config) {
    if (a.dtype != b.dtype || a.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("dot.") + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(b.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(n),
        },
        launch_config);
}

/* Resolve and launch out = alpha * x. */
void launch_scal(
    Context &context,
    const TensorView &x,
    const TensorView &out,
    float alpha,
    const LaunchConfig *launch_config) {
    if (x.dtype != out.dtype || x.dtype != "fp32" || x.elements != out.elements) {
        throw Error(Status::BadArgument);
    }
    if (x.elements > UINT32_MAX) {
        throw Error(Status::OutOfRange);
    }

    context.launch_kernel(
        "scal.fp32",
        {
            KernelArg::device_ptr(x.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::f32(alpha),
            KernelArg::u32(static_cast<std::uint32_t>(x.elements)),
        },
        launch_config);
}

/* Resolve and launch out = alpha * x + y. */
void launch_axpy(
    Context &context,
    const TensorView &x,
    const TensorView &y,
    const TensorView &out,
    float alpha,
    const LaunchConfig *launch_config) {
    if (x.dtype != y.dtype || x.dtype != out.dtype || x.dtype != "fp32" ||
        x.elements != y.elements || x.elements != out.elements) {
        throw Error(Status::BadArgument);
    }
    if (x.elements > UINT32_MAX) {
        throw Error(Status::OutOfRange);
    }

    context.launch_kernel(
        "axpy.fp32",
        {
            KernelArg::device_ptr(x.addr),
            KernelArg::device_ptr(y.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::f32(alpha),
            KernelArg::u32(static_cast<std::uint32_t>(x.elements)),
        },
        launch_config);
}

/* Resolve and launch out = beta * self + alpha * (a @ b). */
void launch_addmm(
    Context &context,
    const TensorView &self,
    const TensorView &a,
    const TensorView &b,
    const TensorView &out,
    float beta,
    float alpha,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    const LaunchConfig *launch_config) {
    if (self.dtype != a.dtype || self.dtype != b.dtype || self.dtype != out.dtype ||
        self.dtype != "fp32") {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        "addmm.fp32",
        {
            KernelArg::device_ptr(self.addr),
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(b.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::f32(beta),
            KernelArg::f32(alpha),
            KernelArg::u32(m),
            KernelArg::u32(n),
            KernelArg::u32(k),
        },
        launch_config);
}

/* Resolve and launch a row-major linear layer kernel without bias. */
void launch_linear(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t out_features,
    std::uint32_t in_features,
    const LaunchConfig *launch_config
) {
    if (input.dtype != weight.dtype || input.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("linear.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(out_features),
            KernelArg::u32(in_features),
        },
        launch_config);
}

/* Resolve and launch a row-major linear layer kernel with bias*/
void launch_linear_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t out_features,
    std::uint32_t in_features,
    const LaunchConfig *launch_config
){
    if (input.dtype != weight.dtype || input.dtype != out.dtype 
        || input.dtype != bias.dtype || weight.dtype != bias.dtype) {
        throw Error(Status::BadArgument);
    }
    context.launch_kernel(
        std::string("linear_bias.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(bias.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(out_features),
            KernelArg::u32(in_features),
        },
        launch_config);
}

/* Resolve and launch_conv1d without bias*/
void launch_conv1d(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total, 
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels, 
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t kernel_width,
    std::uint32_t stride,
    std::uint32_t padding,
    const LaunchConfig *launch_config
){
    if (input.dtype != weight.dtype || input.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("conv1d.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(batch_size),
            KernelArg::u32(in_channels),
            KernelArg::u32(out_channels),
            KernelArg::u32(input_width),
            KernelArg::u32(output_width),
            KernelArg::u32(kernel_width),
            KernelArg::u32(stride),
            KernelArg::u32(padding),
        },
        launch_config);
}

/* Resolve and launch a 1d conv kernel with bias */
void launch_conv1d_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total, 
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels, 
    std::uint32_t input_width,
    std::uint32_t output_width,
    std::uint32_t kernel_width,
    std::uint32_t stride,
    std::uint32_t padding,
    const LaunchConfig *launch_config
){
    if (input.dtype != weight.dtype || input.dtype != out.dtype 
        || input.dtype != bias.dtype || weight.dtype != bias.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("conv1d_bias.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(bias.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(batch_size),
            KernelArg::u32(in_channels),
            KernelArg::u32(out_channels),
            KernelArg::u32(input_width),
            KernelArg::u32(output_width),
            KernelArg::u32(kernel_width),
            KernelArg::u32(stride),
            KernelArg::u32(padding),
        },
        launch_config);
}

/* Resolve and launch a 2d conv kernel without bias */
void launch_conv2d(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels,
    std::uint32_t input_height,
    std::uint32_t input_width,
    std::uint32_t output_height,
    std::uint32_t output_width,
    std::uint32_t kernel_height,
    std::uint32_t kernel_width,
    std::uint32_t stride_h,
    std::uint32_t stride_w,
    std::uint32_t padding_h,
    std::uint32_t padding_w,
    const LaunchConfig *launch_config
) {
    if (input.dtype != weight.dtype || input.dtype != out.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("conv2d.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(batch_size),
            KernelArg::u32(in_channels),
            KernelArg::u32(out_channels),
            KernelArg::u32(input_height),
            KernelArg::u32(input_width),
            KernelArg::u32(output_height),
            KernelArg::u32(output_width),
            KernelArg::u32(kernel_height),
            KernelArg::u32(kernel_width),
            KernelArg::u32(stride_h),
            KernelArg::u32(stride_w),
            KernelArg::u32(padding_h),
            KernelArg::u32(padding_w),
        },
        launch_config);
}

/* Resolve and launch a 2d conv kernel with bias */
void launch_conv2d_bias(
    Context &context,
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    const TensorView &out,
    std::uint32_t total,
    std::uint32_t batch_size,
    std::uint32_t in_channels,
    std::uint32_t out_channels,
    std::uint32_t input_height,
    std::uint32_t input_width,
    std::uint32_t output_height,
    std::uint32_t output_width,
    std::uint32_t kernel_height,
    std::uint32_t kernel_width,
    std::uint32_t stride_h,
    std::uint32_t stride_w,
    std::uint32_t padding_h,
    std::uint32_t padding_w,
    const LaunchConfig *launch_config
) {
    if (input.dtype != weight.dtype || input.dtype != out.dtype ||
        input.dtype != bias.dtype || weight.dtype != bias.dtype) {
        throw Error(Status::BadArgument);
    }

    context.launch_kernel(
        std::string("conv2d_bias.") + std::string(input.dtype),
        {
            KernelArg::device_ptr(input.addr),
            KernelArg::device_ptr(weight.addr),
            KernelArg::device_ptr(bias.addr),
            KernelArg::device_ptr(out.addr),
            KernelArg::u32(total),
            KernelArg::u32(batch_size),
            KernelArg::u32(in_channels),
            KernelArg::u32(out_channels),
            KernelArg::u32(input_height),
            KernelArg::u32(input_width),
            KernelArg::u32(output_height),
            KernelArg::u32(output_width),
            KernelArg::u32(kernel_height),
            KernelArg::u32(kernel_width),
            KernelArg::u32(stride_h),
            KernelArg::u32(stride_w),
            KernelArg::u32(padding_h),
            KernelArg::u32(padding_w),
        },
        launch_config);
}

} // namespace minigpu::kernels
