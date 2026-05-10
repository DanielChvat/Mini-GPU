#include "minigpu_kernels.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
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

/* Parse the runtime kernel manifest. */
std::vector<PrecompiledKernel> parse_manifest(std::string_view manifest_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(std::string(manifest_path));
    } catch (const YAML::Exception &) {
        throw Error(Status::NotFound);
    }

    const YAML::Node kernel_nodes = root["kernels"];
    if (!kernel_nodes || !kernel_nodes.IsSequence()) {
        throw Error(Status::BadArgument);
    }

    std::vector<PrecompiledKernel> kernels;
    for (const YAML::Node &entry : kernel_nodes) {
        if (!entry.IsMap()) {
            throw Error(Status::BadArgument);
        }

        PrecompiledKernel kernel;
        kernel.name = entry["name"].as<std::string>();
        kernel.op = entry["op"].as<std::string>();
        kernel.dtype = entry["dtype"].as<std::string>();
        const std::string file = entry["file"].as<std::string>();
        kernel.program_bytes = read_kernel_file(resolve_artifact_path(manifest_path, file));
        kernel.program_addr = yaml_u32(entry["program_addr"], 0);
        kernel.base_pc = yaml_u32(entry["base_pc"], 0);
        kernel.default_grid_dim = yaml_u32(entry["grid_dim"], 1);
        kernel.default_block_dim = yaml_u32(entry["block_dim"], 4);
        kernel.default_active_mask = yaml_u32(entry["active_mask"], 0xffffffffu);
        kernel.default_timeout_ms = yaml_u32(entry["timeout_ms"], 0);

        if (kernel.name.empty() || kernel.op.empty() ||
            kernel.dtype.empty() || kernel.program_bytes.empty()) {
            throw Error(Status::BadArgument);
        }
        kernels.push_back(std::move(kernel));
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

    context.launch_kernel(
        std::string(op) + "." + std::string(a.dtype),
        {
            KernelArg::device_ptr(a.addr),
            KernelArg::device_ptr(b.addr),
            KernelArg::device_ptr(out.addr),
        },
        launch_config);
}

} // namespace minigpu::kernels
