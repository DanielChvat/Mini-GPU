#include "minigpu_torch.hpp"

#include "elementwise.hpp"
#include "minigpu_kernels.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace minigpu::torch_backend {

namespace {

/* Throw a consistent message for operators that are intentionally skeletons. */
[[noreturn]] void unimplemented_op(const char *name) {
    throw std::runtime_error(
        std::string("Mini-GPU PyTorch op is a stub: ") + name);
}

std::size_t tensor_nbytes(const at::Tensor &tensor) {
    return static_cast<std::size_t>(tensor.numel()) *
           static_cast<std::size_t>(tensor.element_size());
}

void require_minigpu_contiguous(const at::Tensor &tensor, const char *op_name) {
    if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error(std::string(op_name) + " requires a Mini-GPU tensor");
    }
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " requires a contiguous tensor");
    }
}

at::Tensor to_cpu_tensor(const at::Tensor &tensor, const char *op_name) {
    require_minigpu_contiguous(tensor, op_name);
    auto cpu = at::empty(tensor.sizes(), tensor.options().device(c10::DeviceType::CPU));
    runtime_context().copy_from_device(cpu.data_ptr(), device_address(tensor), tensor_nbytes(tensor));
    return cpu;
}

at::Tensor to_minigpu_tensor(const at::Tensor &cpu, const at::TensorOptions &options) {
    auto contiguous = cpu.contiguous();
    auto out = at::empty(contiguous.sizes(), options.dtype(contiguous.scalar_type()));
    runtime_context().copy_to_device(
        device_address(out), contiguous.data_ptr(), tensor_nbytes(contiguous));
    return out;
}

at::Tensor run_fp32_unary_kernel(
    const at::Tensor &a,
    const char *kernel_op,
    const char *op_name) {
    require_minigpu_contiguous(a, op_name);
    if (a.scalar_type() != at::kFloat) {
        throw std::runtime_error(std::string(op_name) + " on Mini-GPU currently supports fp32 only");
    }
    return detail::run_unary_kernel(a, kernel_op, op_name);
}

std::uint32_t checked_u32(std::int64_t value, const char *name) {
    if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(std::string(name) + " is out of range for Mini-GPU");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t sym_u32(c10::SymInt value, const char *name) {
    return checked_u32(value.expect_int(), name);
}

std::uint32_t sym_array_u32(c10::SymIntArrayRef values, std::size_t index, const char *name) {
    if (values.size() <= static_cast<std::int64_t>(index)) {
        throw std::runtime_error(std::string(name) + " has too few dimensions");
    }
    return sym_u32(values[index], name);
}

bool sym_array_all(c10::SymIntArrayRef values, std::int64_t expected) {
    for (const auto &value : values) {
        if (value.expect_int() != expected) {
            return false;
        }
    }
    return true;
}

bool int_array_all(at::IntArrayRef values, std::int64_t expected) {
    for (const auto value : values) {
        if (value != expected) {
            return false;
        }
    }
    return true;
}

std::vector<std::int64_t> scalar_or_keepdim_shape(const at::Tensor &tensor, bool keepdim) {
    if (!keepdim) {
        return {};
    }
    return std::vector<std::int64_t>(static_cast<std::size_t>(tensor.dim()), 1);
}

bool dims_cover_all(at::OptionalIntArrayRef dims, std::int64_t rank) {
    if (!dims.has_value()) {
        return true;
    }
    if (dims->empty()) {
        return true;
    }
    if (static_cast<std::int64_t>(dims->size()) != rank) {
        return false;
    }
    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (auto dim : *dims) {
        if (dim < 0) {
            dim += rank;
        }
        if (dim < 0 || dim >= rank || seen[static_cast<std::size_t>(dim)]) {
            return false;
        }
        seen[static_cast<std::size_t>(dim)] = true;
    }
    return true;
}

bool dims_cover_all(at::IntArrayRef dims, std::int64_t rank) {
    if (dims.empty()) {
        return true;
    }
    if (static_cast<std::int64_t>(dims.size()) != rank) {
        return false;
    }
    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (auto dim : dims) {
        if (dim < 0) {
            dim += rank;
        }
        if (dim < 0 || dim >= rank || seen[static_cast<std::size_t>(dim)]) {
            return false;
        }
        seen[static_cast<std::size_t>(dim)] = true;
    }
    return true;
}

std::array<std::uint32_t, 2> pair_u32_or(
    at::IntArrayRef values,
    std::uint32_t fallback_h,
    std::uint32_t fallback_w,
    const char *name) {
    if (values.empty()) {
        return {fallback_h, fallback_w};
    }
    if (values.size() == 1) {
        auto value = checked_u32(values[0], name);
        return {value, value};
    }
    if (values.size() == 2) {
        return {checked_u32(values[0], name), checked_u32(values[1], name)};
    }
    throw std::runtime_error(std::string(name) + " expects one or two values");
}

std::string_view supported_kernel_dtype(const at::Tensor &tensor, const char *op_name) {
    if (tensor.scalar_type() == at::kInt) {
        return "i32";
    }
    if (tensor.scalar_type() == at::kFloat) {
        return "fp32";
    }
    throw std::runtime_error(std::string(op_name) + " on Mini-GPU currently supports int32 and fp32");
}

void require_fp32_minigpu_contiguous(const at::Tensor &tensor, const char *op_name) {
    require_minigpu_contiguous(tensor, op_name);
    if (tensor.scalar_type() != at::kFloat) {
        throw std::runtime_error(std::string(op_name) + " on Mini-GPU currently supports fp32 only");
    }
}

at::Tensor run_global_reduction_kernel(
    const at::Tensor &a,
    const char *kernel_name,
    const char *op_name,
    bool keepdim) {
    require_fp32_minigpu_contiguous(a, op_name);
    if (a.numel() <= 0) {
        throw std::runtime_error(std::string(op_name) + " on Mini-GPU requires non-empty input");
    }
    auto out = at::empty(scalar_or_keepdim_shape(a, keepdim), a.options());
    runtime_context().launch_kernel(
        kernel_name,
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(a.numel(), "reduction input size")),
        });
    return out;
}

at::Tensor run_mean_kernel(
    const at::Tensor &a,
    const char *op_name,
    bool keepdim) {
    require_fp32_minigpu_contiguous(a, op_name);
    if (a.numel() <= 0) {
        throw std::runtime_error(std::string(op_name) + " on Mini-GPU requires non-empty input");
    }
    auto out = at::empty(scalar_or_keepdim_shape(a, keepdim), a.options());
    const auto count = checked_u32(a.numel(), "mean input size");
    runtime_context().launch_kernel(
        "mean.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(count),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(count)),
        });
    return out;
}

} // namespace

template <typename T>
at::Tensor transpose_weight_to_device(const at::Tensor &weight) {
    const auto out_features = weight.size(0);
    const auto in_features = weight.size(1);
    std::vector<T> host(static_cast<std::size_t>(weight.numel()));
    std::vector<T> transposed(host.size());

    runtime_context().copy_from_device(
        host.data(), device_address(weight), host.size() * sizeof(T));

    for (std::int64_t row = 0; row < out_features; ++row) {
        for (std::int64_t col = 0; col < in_features; ++col) {
            transposed[static_cast<std::size_t>(col * out_features + row)] =
                host[static_cast<std::size_t>(row * in_features + col)];
        }
    }

    auto out = at::empty({in_features, out_features}, weight.options());
    runtime_context().copy_to_device(
        device_address(out), transposed.data(), transposed.size() * sizeof(T));
    return out;
}

at::Tensor add_tensor(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha) {
    require_minigpu_contiguous(a, "aten::add.Tensor self");
    require_minigpu_contiguous(b, "aten::add.Tensor other");
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error("aten::add.Tensor on Mini-GPU requires matching shapes");
    }
    if (a.scalar_type() != b.scalar_type()) {
        throw std::runtime_error("aten::add.Tensor on Mini-GPU requires matching dtypes");
    }
    if (alpha.toDouble() != 1.0) {
        if (a.scalar_type() != at::kFloat) {
            throw std::runtime_error("aten::add.Tensor on Mini-GPU currently supports alpha!=1 for fp32 only");
        }
        return axpy(b, a, alpha);
    }

    return detail::run_vector_add_kernel(a, b, "aten::add.Tensor");
}

at::Tensor &add_tensor_(at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha) {
    require_minigpu_contiguous(a, "aten::add_.Tensor self");
    require_minigpu_contiguous(b, "aten::add_.Tensor other");
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error("aten::add_.Tensor on Mini-GPU requires matching shapes");
    }
    if (a.scalar_type() != b.scalar_type()) {
        throw std::runtime_error("aten::add_.Tensor on Mini-GPU requires matching dtypes");
    }
    auto updated = add_tensor(a, b, alpha);
    copy_(a, updated, false);
    return a;
}

at::Tensor vector_add(const at::Tensor &a, const at::Tensor &b) {
    return detail::run_vector_add_kernel(a, b, "minigpu::vector_add");
}

at::Tensor mul_tensor(const at::Tensor &a, const at::Tensor &b) {
    return detail::run_binary_kernel(a, b, "mul", "aten::mul.Tensor");
}

at::Tensor div_tensor(const at::Tensor &a, const at::Tensor &b) {
    if (a.device().type() != c10::DeviceType::PrivateUse1 ||
        b.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("aten::div.Tensor requires Mini-GPU tensors");
    }
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error("aten::div.Tensor on Mini-GPU requires matching shapes");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("aten::div.Tensor on Mini-GPU requires contiguous tensors");
    }
    if (a.scalar_type() != at::kFloat || b.scalar_type() != at::kFloat) {
        throw std::runtime_error("aten::div.Tensor on Mini-GPU currently supports fp32 only");
    }

    auto out = at::empty_like(a);
    minigpu::kernels::launch_elementwise_binary(
        runtime_context(),
        "div",
        minigpu::kernels::TensorView{
            device_address(a), static_cast<std::size_t>(a.numel()), "fp32"},
        minigpu::kernels::TensorView{
            device_address(b), static_cast<std::size_t>(b.numel()), "fp32"},
        minigpu::kernels::TensorView{
            device_address(out), static_cast<std::size_t>(out.numel()), "fp32"});

    return out;
}

at::Tensor mm(const at::Tensor &a, const at::Tensor &b) {
    if (a.device().type() != c10::DeviceType::PrivateUse1 ||
        b.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("aten::mm requires Mini-GPU tensors");
    }
    if (a.dim() != 2 || b.dim() != 2) {
        throw std::runtime_error("aten::mm on Mini-GPU requires 2D tensors");
    }
    if (a.size(1) != b.size(0)) {
        throw std::runtime_error("aten::mm on Mini-GPU requires compatible shapes");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error("aten::mm on Mini-GPU requires contiguous tensors");
    }
    if (a.scalar_type() != b.scalar_type()) {
        throw std::runtime_error("aten::mm on Mini-GPU requires matching dtypes");
    }

    std::string_view dtype;
    if (a.scalar_type() == at::kInt) {
        dtype = "i32";
    } else if (a.scalar_type() == at::kFloat) {
        dtype = "fp32";
    } else {
        throw std::runtime_error("aten::mm on Mini-GPU currently supports int32 and fp32");
    }

    auto out = at::empty({a.size(0), b.size(1)}, a.options());
    const auto m = static_cast<std::uint32_t>(a.size(0));
    const auto n = static_cast<std::uint32_t>(b.size(1));
    const auto k = static_cast<std::uint32_t>(a.size(1));

    minigpu::kernels::launch_matmul(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(a), static_cast<std::size_t>(a.numel()), dtype},
        minigpu::kernels::TensorView{device_address(b), static_cast<std::size_t>(b.numel()), dtype},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), dtype},
        m,
        n,
        k);

    return out;
}

at::Tensor mv(const at::Tensor &a, const at::Tensor &x) {
    require_minigpu_contiguous(a, "aten::mv input");
    require_minigpu_contiguous(x, "aten::mv vec");
    if (a.dim() != 2 || x.dim() != 1) {
        throw std::runtime_error("aten::mv on Mini-GPU requires a 2D matrix and 1D vector");
    }
    if (a.size(1) != x.size(0)) {
        throw std::runtime_error("aten::mv on Mini-GPU requires compatible shapes");
    }
    if (a.scalar_type() != x.scalar_type()) {
        throw std::runtime_error("aten::mv on Mini-GPU requires matching dtypes");
    }

    std::string_view dtype = supported_kernel_dtype(a, "aten::mv");
    auto out = at::empty({a.size(0)}, a.options());

    minigpu::kernels::launch_gemv(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(a), static_cast<std::size_t>(a.numel()), dtype},
        minigpu::kernels::TensorView{device_address(x), static_cast<std::size_t>(x.numel()), dtype},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), dtype},
        static_cast<std::uint32_t>(a.size(0)),
        static_cast<std::uint32_t>(a.size(1)));

    return out;
}

at::Tensor dot(const at::Tensor &a, const at::Tensor &b) {
    require_minigpu_contiguous(a, "aten::dot input");
    require_minigpu_contiguous(b, "aten::dot input");
    if (a.dim() != 1 || b.dim() != 1) {
        throw std::runtime_error("aten::dot on Mini-GPU requires 1D tensors");
    }
    if (a.size(0) != b.size(0)) {
        throw std::runtime_error("aten::dot on Mini-GPU requires matching lengths");
    }
    if (a.scalar_type() != b.scalar_type()) {
        throw std::runtime_error("aten::dot on Mini-GPU requires matching dtypes");
    }

    std::string_view dtype = supported_kernel_dtype(a, "aten::dot");
    auto out = at::empty({}, a.options());

    minigpu::kernels::launch_dot(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(a), static_cast<std::size_t>(a.numel()), dtype},
        minigpu::kernels::TensorView{device_address(b), static_cast<std::size_t>(b.numel()), dtype},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), dtype},
        static_cast<std::uint32_t>(a.size(0)));

    return out;
}

at::Tensor scal(const at::Tensor &x, const at::Scalar &alpha) {
    require_minigpu_contiguous(x, "minigpu::scal");
    if (x.scalar_type() != at::kFloat) {
        throw std::runtime_error("minigpu::scal currently supports fp32 only");
    }

    auto out = at::empty_like(x);
    minigpu::kernels::launch_scal(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(x), static_cast<std::size_t>(x.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), "fp32"},
        static_cast<float>(alpha.toDouble()));
    return out;
}

at::Tensor axpy(const at::Tensor &x, const at::Tensor &y, const at::Scalar &alpha) {
    require_minigpu_contiguous(x, "minigpu::axpy x");
    require_minigpu_contiguous(y, "minigpu::axpy y");
    if (x.sizes() != y.sizes()) {
        throw std::runtime_error("minigpu::axpy requires matching shapes");
    }
    if (x.scalar_type() != y.scalar_type()) {
        throw std::runtime_error("minigpu::axpy requires matching dtypes");
    }
    if (x.scalar_type() != at::kFloat) {
        throw std::runtime_error("minigpu::axpy currently supports fp32 only");
    }

    auto out = at::empty_like(x);
    minigpu::kernels::launch_axpy(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(x), static_cast<std::size_t>(x.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(y), static_cast<std::size_t>(y.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), "fp32"},
        static_cast<float>(alpha.toDouble()));
    return out;
}

at::Tensor addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha) {
    require_minigpu_contiguous(self, "aten::addmm self");
    require_minigpu_contiguous(mat1, "aten::addmm mat1");
    require_minigpu_contiguous(mat2, "aten::addmm mat2");
    if (self.dim() != 2 || mat1.dim() != 2 || mat2.dim() != 2) {
        throw std::runtime_error("aten::addmm on Mini-GPU currently requires 2D tensors");
    }
    if (mat1.size(1) != mat2.size(0)) {
        throw std::runtime_error("aten::addmm on Mini-GPU requires compatible matmul shapes");
    }
    if (self.size(0) != mat1.size(0) || self.size(1) != mat2.size(1)) {
        throw std::runtime_error("aten::addmm on Mini-GPU currently requires self shape [mat1.rows, mat2.cols]");
    }
    if (self.scalar_type() != mat1.scalar_type() || self.scalar_type() != mat2.scalar_type()) {
        throw std::runtime_error("aten::addmm on Mini-GPU requires matching dtypes");
    }
    if (self.scalar_type() != at::kFloat) {
        throw std::runtime_error("aten::addmm on Mini-GPU currently supports fp32 only");
    }

    auto out = at::empty_like(self);
    minigpu::kernels::launch_addmm(
        runtime_context(),
        minigpu::kernels::TensorView{device_address(self), static_cast<std::size_t>(self.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(mat1), static_cast<std::size_t>(mat1.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(mat2), static_cast<std::size_t>(mat2.numel()), "fp32"},
        minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), "fp32"},
        static_cast<float>(beta.toDouble()),
        static_cast<float>(alpha.toDouble()),
        static_cast<std::uint32_t>(mat1.size(0)),
        static_cast<std::uint32_t>(mat2.size(1)),
        static_cast<std::uint32_t>(mat1.size(1)));

    return out;
}

at::Tensor linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const ::std::optional<at::Tensor> &bias) {

    const bool has_bias = bias.has_value() && bias->defined();
    if (input.device().type() != c10::DeviceType::PrivateUse1 ||
        weight.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("aten::linear requires Mini-GPU tensors");
    }
    if (input.dim() != 1 && input.dim() != 2) {
        throw std::runtime_error("aten::linear on Mini-GPU currently supports 1D or 2D input");
    }
    if (weight.dim() != 2) {
        throw std::runtime_error("aten::linear on Mini-GPU requires a 2D weight tensor");
    }
    if (!input.is_contiguous() || !weight.is_contiguous()) {
        throw std::runtime_error("aten::linear on Mini-GPU requires contiguous tensors");
    }
    if (input.scalar_type() != weight.scalar_type()) {
        throw std::runtime_error("aten::linear on Mini-GPU requires matching input and weight dtypes");
    }
    if (input.size(input.dim() - 1) != weight.size(1)) {
        throw std::runtime_error("aten::linear on Mini-GPU requires matching in_features");
    }

    std::string_view dtype = supported_kernel_dtype(input, "aten::linear");

    if (has_bias) {
        require_minigpu_contiguous(*bias, "aten::linear bias");
        if (bias->dim() != 1 || bias->size(0) != weight.size(0)) {
            throw std::runtime_error("aten::linear on Mini-GPU requires bias shape [out_features]");
        }
        if (bias->scalar_type() != input.scalar_type()) {
            throw std::runtime_error("aten::linear on Mini-GPU requires matching bias dtype");
        }
    }

    at::Tensor input_2d = input.dim() == 1 ? input.view({1, input.size(0)}) : input;
    auto out_2d = at::empty({input_2d.size(0), weight.size(0)}, input.options());

    if (has_bias) {
        minigpu::kernels::launch_linear_bias(
            runtime_context(),
            minigpu::kernels::TensorView{
                device_address(input_2d), static_cast<std::size_t>(input_2d.numel()), dtype},
            minigpu::kernels::TensorView{
                device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
            minigpu::kernels::TensorView{
                device_address(*bias), static_cast<std::size_t>(bias->numel()), dtype},
            minigpu::kernels::TensorView{
                device_address(out_2d), static_cast<std::size_t>(out_2d.numel()), dtype},
            static_cast<std::uint32_t>(out_2d.numel()),
            static_cast<std::uint32_t>(weight.size(0)),
            static_cast<std::uint32_t>(weight.size(1)));
    } else {
        minigpu::kernels::launch_linear(
            runtime_context(),
            minigpu::kernels::TensorView{
                device_address(input_2d), static_cast<std::size_t>(input_2d.numel()), dtype},
            minigpu::kernels::TensorView{
                device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
            minigpu::kernels::TensorView{
                device_address(out_2d), static_cast<std::size_t>(out_2d.numel()), dtype},
            static_cast<std::uint32_t>(out_2d.numel()),
            static_cast<std::uint32_t>(weight.size(0)),
            static_cast<std::uint32_t>(weight.size(1)));
    }
    return input.dim() == 1 ? out_2d.view({weight.size(0)}) : out_2d;
}

at::Tensor convolution(
    const at::Tensor &input,
    const at::Tensor &weight,
    const ::std::optional<at::Tensor> &bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups) {
    require_minigpu_contiguous(input, "aten::convolution input");
    require_minigpu_contiguous(weight, "aten::convolution weight");
    if (input.scalar_type() != weight.scalar_type()) {
        throw std::runtime_error("aten::convolution on Mini-GPU requires matching input and weight dtypes");
    }
    if (transposed) {
        throw std::runtime_error("aten::convolution on Mini-GPU does not support transposed convolution yet");
    }
    if (groups.expect_int() != 1) {
        throw std::runtime_error("aten::convolution on Mini-GPU currently supports groups=1 only");
    }
    if (!sym_array_all(dilation, 1)) {
        throw std::runtime_error("aten::convolution on Mini-GPU currently supports dilation=1 only");
    }
    if (!sym_array_all(output_padding, 0)) {
        throw std::runtime_error("aten::convolution on Mini-GPU requires output_padding=0");
    }
    if (input.dim() != 3 && input.dim() != 4) {
        throw std::runtime_error("aten::convolution on Mini-GPU currently supports NCL and NCHW inputs");
    }
    if (weight.dim() != input.dim()) {
        throw std::runtime_error("aten::convolution on Mini-GPU weight rank must match input rank");
    }
    if (input.size(1) != weight.size(1)) {
        throw std::runtime_error("aten::convolution on Mini-GPU requires matching input channels");
    }

    std::string_view dtype = supported_kernel_dtype(input, "aten::convolution");
    const bool has_bias = bias.has_value() && bias->defined();
    if (has_bias) {
        require_minigpu_contiguous(*bias, "aten::convolution bias");
        if (bias->dim() != 1 || bias->size(0) != weight.size(0)) {
            throw std::runtime_error("aten::convolution on Mini-GPU requires bias shape [out_channels]");
        }
        if (bias->scalar_type() != input.scalar_type()) {
            throw std::runtime_error("aten::convolution on Mini-GPU requires matching bias dtype");
        }
    }

    const auto batch_size = checked_u32(input.size(0), "batch_size");
    const auto in_channels = checked_u32(input.size(1), "in_channels");
    const auto out_channels = checked_u32(weight.size(0), "out_channels");

    if (input.dim() == 3) {
        if (stride.size() != 1 || padding.size() != 1 || dilation.size() != 1 || output_padding.size() != 1) {
            throw std::runtime_error("aten::convolution 1D expects one stride/padding/dilation value");
        }
        const auto input_width = checked_u32(input.size(2), "input_width");
        const auto kernel_width = checked_u32(weight.size(2), "kernel_width");
        const auto stride_w = sym_array_u32(stride, 0, "stride");
        const auto padding_w = sym_array_u32(padding, 0, "padding");
        if (padding_w != 0) {
            throw std::runtime_error("aten::convolution 1D on Mini-GPU currently supports padding=0 only");
        }
        if (stride_w == 0) {
            throw std::runtime_error("aten::convolution 1D on Mini-GPU requires nonzero stride");
        }
        const auto output_width =
            checked_u32((input.size(2) + 2 * static_cast<std::int64_t>(padding_w) -
                         static_cast<std::int64_t>(kernel_width)) /
                            static_cast<std::int64_t>(stride_w) +
                        1,
                        "output_width");
        auto out = at::empty({input.size(0), weight.size(0), output_width}, input.options());
        auto out_view = minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), dtype};

        if (has_bias) {
            minigpu::kernels::launch_conv1d_bias(
                runtime_context(),
                minigpu::kernels::TensorView{device_address(input), static_cast<std::size_t>(input.numel()), dtype},
                minigpu::kernels::TensorView{device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
                minigpu::kernels::TensorView{device_address(*bias), static_cast<std::size_t>(bias->numel()), dtype},
                out_view,
                static_cast<std::uint32_t>(out.numel()),
                batch_size,
                in_channels,
                out_channels,
                input_width,
                output_width,
                kernel_width,
                stride_w,
                padding_w);
        } else {
            minigpu::kernels::launch_conv1d(
                runtime_context(),
                minigpu::kernels::TensorView{device_address(input), static_cast<std::size_t>(input.numel()), dtype},
                minigpu::kernels::TensorView{device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
                out_view,
                static_cast<std::uint32_t>(out.numel()),
                batch_size,
                in_channels,
                out_channels,
                input_width,
                output_width,
                kernel_width,
                stride_w,
                padding_w);
        }
        return out;
    }

    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 || output_padding.size() != 2) {
        throw std::runtime_error("aten::convolution 2D expects two stride/padding/dilation values");
    }
    const auto input_height = checked_u32(input.size(2), "input_height");
    const auto input_width = checked_u32(input.size(3), "input_width");
    const auto kernel_height = checked_u32(weight.size(2), "kernel_height");
    const auto kernel_width = checked_u32(weight.size(3), "kernel_width");
    const auto stride_h = sym_array_u32(stride, 0, "stride");
    const auto stride_w = sym_array_u32(stride, 1, "stride");
    const auto padding_h = sym_array_u32(padding, 0, "padding");
    const auto padding_w = sym_array_u32(padding, 1, "padding");
    if (padding_h != 0 || padding_w != 0) {
        throw std::runtime_error("aten::convolution 2D on Mini-GPU currently supports padding=0 only");
    }
    if (stride_h == 0 || stride_w == 0) {
        throw std::runtime_error("aten::convolution 2D on Mini-GPU requires nonzero stride");
    }
    const auto output_height =
        checked_u32((input.size(2) + 2 * static_cast<std::int64_t>(padding_h) -
                     static_cast<std::int64_t>(kernel_height)) /
                        static_cast<std::int64_t>(stride_h) +
                    1,
                    "output_height");
    const auto output_width =
        checked_u32((input.size(3) + 2 * static_cast<std::int64_t>(padding_w) -
                     static_cast<std::int64_t>(kernel_width)) /
                        static_cast<std::int64_t>(stride_w) +
                    1,
                    "output_width");
    auto out = at::empty({input.size(0), weight.size(0), output_height, output_width}, input.options());
    auto out_view = minigpu::kernels::TensorView{device_address(out), static_cast<std::size_t>(out.numel()), dtype};

    if (has_bias) {
        minigpu::kernels::launch_conv2d_bias(
            runtime_context(),
            minigpu::kernels::TensorView{device_address(input), static_cast<std::size_t>(input.numel()), dtype},
            minigpu::kernels::TensorView{device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
            minigpu::kernels::TensorView{device_address(*bias), static_cast<std::size_t>(bias->numel()), dtype},
            out_view,
            static_cast<std::uint32_t>(out.numel()),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w);
    } else {
        minigpu::kernels::launch_conv2d(
            runtime_context(),
            minigpu::kernels::TensorView{device_address(input), static_cast<std::size_t>(input.numel()), dtype},
            minigpu::kernels::TensorView{device_address(weight), static_cast<std::size_t>(weight.numel()), dtype},
            out_view,
            static_cast<std::uint32_t>(out.numel()),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w);
    }
    return out;
}

at::Tensor relu(const at::Tensor &a) {
    return detail::run_relu_kernel(a, "aten::relu");
}

at::Tensor exp(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "exp", "aten::exp");
}

at::Tensor log(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "log", "aten::log");
}

at::Tensor log2(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "log2", "aten::log2");
}

at::Tensor log10(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "log10", "aten::log10");
}

at::Tensor sqrt(const at::Tensor &a) {
    if (a.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("aten::sqrt requires a Mini-GPU tensor");
    }
    if (!a.is_contiguous()) {
        throw std::runtime_error("aten::sqrt on Mini-GPU requires a contiguous tensor");
    }
    if (a.scalar_type() != at::kFloat) {
        throw std::runtime_error("aten::sqrt on Mini-GPU currently supports fp32 only");
    }

    auto out = at::empty_like(a);
    minigpu::kernels::launch_elementwise_unary(
        runtime_context(),
        "sqrt",
        minigpu::kernels::TensorView{
            device_address(a), static_cast<std::size_t>(a.numel()), "fp32"},
        minigpu::kernels::TensorView{
            device_address(out), static_cast<std::size_t>(out.numel()), "fp32"});

    return out;
}

at::Tensor reciprocal(const at::Tensor &a) {
    if (a.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error("aten::reciprocal requires a Mini-GPU tensor");
    }
    if (!a.is_contiguous()) {
        throw std::runtime_error("aten::reciprocal on Mini-GPU requires a contiguous tensor");
    }
    if (a.scalar_type() != at::kFloat) {
        throw std::runtime_error("aten::reciprocal on Mini-GPU currently supports fp32 only");
    }

    auto out = at::empty_like(a);
    minigpu::kernels::launch_elementwise_unary(
        runtime_context(),
        "reciprocal",
        minigpu::kernels::TensorView{
            device_address(a), static_cast<std::size_t>(a.numel()), "fp32"},
        minigpu::kernels::TensorView{
            device_address(out), static_cast<std::size_t>(out.numel()), "fp32"});

    return out;
}

at::Tensor pow_tensor_tensor(const at::Tensor &a, const at::Tensor &b) {
    require_minigpu_contiguous(a, "aten::pow.Tensor_Tensor");
    require_minigpu_contiguous(b, "aten::pow.Tensor_Tensor");
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error("aten::pow.Tensor_Tensor requires matching shapes");
    }
    if (a.scalar_type() != at::kFloat || b.scalar_type() != at::kFloat) {
        throw std::runtime_error("aten::pow.Tensor_Tensor on Mini-GPU currently supports fp32 only");
    }
    return to_minigpu_tensor(
        at::pow(to_cpu_tensor(a, "aten::pow.Tensor_Tensor"),
                to_cpu_tensor(b, "aten::pow.Tensor_Tensor")),
        a.options());
}

at::Tensor pow_tensor_scalar(const at::Tensor &a, const at::Scalar &b) {
    return to_minigpu_tensor(
        at::pow(to_cpu_tensor(a, "aten::pow.Tensor_Scalar"), b),
        a.options());
}

at::Tensor sigmoid(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "sigmoid", "aten::sigmoid");
}

at::Tensor tanh(const at::Tensor &a) {
    return run_fp32_unary_kernel(a, "tanh", "aten::tanh");
}

at::Tensor sin(const at::Tensor &a) {
    return to_minigpu_tensor(at::sin(to_cpu_tensor(a, "aten::sin")), a.options());
}

at::Tensor cos(const at::Tensor &a) {
    return to_minigpu_tensor(at::cos(to_cpu_tensor(a, "aten::cos")), a.options());
}

at::Tensor tan(const at::Tensor &a) {
    return to_minigpu_tensor(at::tan(to_cpu_tensor(a, "aten::tan")), a.options());
}

at::Tensor sum(const at::Tensor &a, c10::optional<c10::ScalarType> dtype) {
    if (dtype.has_value() && *dtype != at::kFloat) {
        throw std::runtime_error("aten::sum on Mini-GPU currently supports fp32 output only");
    }
    return run_global_reduction_kernel(a, "sum.fp32", "aten::sum", false);
}

at::Tensor sum_dim(
    const at::Tensor &a,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
    if (!dims_cover_all(dim, a.dim())) {
        throw std::runtime_error("aten::sum.dim_IntList on Mini-GPU currently supports all-dim reductions only");
    }
    if (dtype.has_value() && *dtype != at::kFloat) {
        throw std::runtime_error("aten::sum.dim_IntList on Mini-GPU currently supports fp32 output only");
    }
    return run_global_reduction_kernel(a, "sum.fp32", "aten::sum.dim_IntList", keepdim);
}

at::Tensor mean(const at::Tensor &a, c10::optional<c10::ScalarType> dtype) {
    if (dtype.has_value() && *dtype != at::kFloat) {
        throw std::runtime_error("aten::mean on Mini-GPU currently supports fp32 output only");
    }
    return run_mean_kernel(a, "aten::mean", false);
}

at::Tensor mean_dim(
    const at::Tensor &a,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
    if (!dims_cover_all(dim, a.dim())) {
        throw std::runtime_error("aten::mean.dim on Mini-GPU currently supports all-dim reductions only");
    }
    if (dtype.has_value() && *dtype != at::kFloat) {
        throw std::runtime_error("aten::mean.dim on Mini-GPU currently supports fp32 output only");
    }
    return run_mean_kernel(a, "aten::mean.dim", keepdim);
}

at::Tensor amax(const at::Tensor &a, at::IntArrayRef dim, bool keepdim) {
    if (!dims_cover_all(dim, a.dim())) {
        throw std::runtime_error("aten::amax on Mini-GPU currently supports all-dim reductions only");
    }
    return run_global_reduction_kernel(a, "amax.fp32", "aten::amax", keepdim);
}

at::Tensor amin(const at::Tensor &a, at::IntArrayRef dim, bool keepdim) {
    if (!dims_cover_all(dim, a.dim())) {
        throw std::runtime_error("aten::amin on Mini-GPU currently supports all-dim reductions only");
    }
    return run_global_reduction_kernel(a, "amin.fp32", "aten::amin", keepdim);
}

at::Tensor argmax(const at::Tensor &a, ::std::optional<std::int64_t> dim, bool keepdim) {
    if (dim.has_value()) {
        throw std::runtime_error("aten::argmax on Mini-GPU currently supports dim=None only");
    }
    require_fp32_minigpu_contiguous(a, "aten::argmax");
    auto out = at::empty(scalar_or_keepdim_shape(a, keepdim), a.options().dtype(at::kInt));
    runtime_context().launch_kernel(
        "argmax.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(a.numel(), "argmax input size")),
        });
    return out;
}

at::Tensor argmin(const at::Tensor &a, ::std::optional<std::int64_t> dim, bool keepdim) {
    if (dim.has_value()) {
        throw std::runtime_error("aten::argmin on Mini-GPU currently supports dim=None only");
    }
    require_fp32_minigpu_contiguous(a, "aten::argmin");
    auto out = at::empty(scalar_or_keepdim_shape(a, keepdim), a.options().dtype(at::kInt));
    runtime_context().launch_kernel(
        "argmin.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(a.numel(), "argmin input size")),
        });
    return out;
}

at::Tensor softmax(const at::Tensor &a, std::int64_t dim, c10::optional<c10::ScalarType> dtype) {
    if (dtype.has_value() && *dtype != at::kFloat) {
        throw std::runtime_error("aten::softmax on Mini-GPU currently supports fp32 output only");
    }
    return softmax_impl(a, dim, false);
}

at::Tensor softmax_impl(const at::Tensor &a, std::int64_t dim, bool half_to_float) {
    if (half_to_float) {
        throw std::runtime_error("aten::_softmax on Mini-GPU does not support half_to_float yet");
    }
    require_fp32_minigpu_contiguous(a, "aten::_softmax");
    if (a.dim() == 0) {
        throw std::runtime_error("aten::_softmax on Mini-GPU requires at least 1D input");
    }
    if (dim < 0) {
        dim += a.dim();
    }
    if (dim != a.dim() - 1) {
        throw std::runtime_error("aten::_softmax on Mini-GPU currently supports the last dimension only");
    }
    auto out = at::empty_like(a);
    const auto cols = checked_u32(a.size(dim), "softmax cols");
    const auto rows = checked_u32(a.numel() / a.size(dim), "softmax rows");
    runtime_context().launch_kernel(
        "softmax.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::u32(cols),
        });
    return out;
}

at::Tensor max_pool2d(
    const at::Tensor &a,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
    require_fp32_minigpu_contiguous(a, "aten::max_pool2d");
    if (a.dim() != 4) {
        throw std::runtime_error("aten::max_pool2d on Mini-GPU requires NCHW input");
    }
    auto [kernel_h, kernel_w] = pair_u32_or(kernel_size, 0, 0, "max_pool2d kernel_size");
    auto [stride_h, stride_w] = pair_u32_or(stride, kernel_h, kernel_w, "max_pool2d stride");
    auto [padding_h, padding_w] = pair_u32_or(padding, 0, 0, "max_pool2d padding");
    auto [dilation_h, dilation_w] = pair_u32_or(dilation, 1, 1, "max_pool2d dilation");
    if (padding_h != 0 || padding_w != 0 || dilation_h != 1 || dilation_w != 1 || ceil_mode) {
        throw std::runtime_error("aten::max_pool2d on Mini-GPU currently requires padding=0 dilation=1 ceil_mode=False");
    }
    if (kernel_h != 2 || kernel_w != 2) {
        throw std::runtime_error("aten::max_pool2d on Mini-GPU currently supports kernel_size=2 only");
    }
    const auto out_h = checked_u32((a.size(2) - kernel_h) / stride_h + 1, "max_pool2d output height");
    const auto out_w = checked_u32((a.size(3) - kernel_w) / stride_w + 1, "max_pool2d output width");
    auto out = at::empty({a.size(0), a.size(1), out_h, out_w}, a.options());
    runtime_context().launch_kernel(
        "max_pool2d.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(out.numel(), "max_pool2d output size")),
            minigpu::KernelArg::u32(checked_u32(a.size(1), "max_pool2d channels")),
            minigpu::KernelArg::u32(checked_u32(a.size(2), "max_pool2d input height")),
            minigpu::KernelArg::u32(checked_u32(a.size(3), "max_pool2d input width")),
            minigpu::KernelArg::u32(out_h),
            minigpu::KernelArg::u32(out_w),
            minigpu::KernelArg::u32(stride_h),
            minigpu::KernelArg::u32(stride_w),
        });
    return out;
}

at::Tensor avg_pool2d(
    const at::Tensor &a,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    ::std::optional<std::int64_t> divisor_override) {
    require_fp32_minigpu_contiguous(a, "aten::avg_pool2d");
    if (a.dim() != 4) {
        throw std::runtime_error("aten::avg_pool2d on Mini-GPU requires NCHW input");
    }
    auto [kernel_h, kernel_w] = pair_u32_or(kernel_size, 0, 0, "avg_pool2d kernel_size");
    auto [stride_h, stride_w] = pair_u32_or(stride, kernel_h, kernel_w, "avg_pool2d stride");
    auto [padding_h, padding_w] = pair_u32_or(padding, 0, 0, "avg_pool2d padding");
    if (padding_h != 0 || padding_w != 0 || ceil_mode || !count_include_pad || divisor_override.has_value()) {
        throw std::runtime_error("aten::avg_pool2d on Mini-GPU currently requires padding=0 ceil_mode=False count_include_pad=True and no divisor_override");
    }
    if (kernel_h != 2 || kernel_w != 2) {
        throw std::runtime_error("aten::avg_pool2d on Mini-GPU currently supports kernel_size=2 only");
    }
    const auto out_h = checked_u32((a.size(2) - kernel_h) / stride_h + 1, "avg_pool2d output height");
    const auto out_w = checked_u32((a.size(3) - kernel_w) / stride_w + 1, "avg_pool2d output width");
    auto out = at::empty({a.size(0), a.size(1), out_h, out_w}, a.options());
    runtime_context().launch_kernel(
        "avg_pool2d.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(out.numel(), "avg_pool2d output size")),
            minigpu::KernelArg::u32(checked_u32(a.size(1), "avg_pool2d channels")),
            minigpu::KernelArg::u32(checked_u32(a.size(2), "avg_pool2d input height")),
            minigpu::KernelArg::u32(checked_u32(a.size(3), "avg_pool2d input width")),
            minigpu::KernelArg::u32(out_h),
            minigpu::KernelArg::u32(out_w),
            minigpu::KernelArg::u32(stride_h),
            minigpu::KernelArg::u32(stride_w),
        });
    return out;
}

at::Tensor adaptive_avg_pool2d(const at::Tensor &a, c10::SymIntArrayRef output_size) {
    require_fp32_minigpu_contiguous(a, "aten::adaptive_avg_pool2d");
    if (a.dim() != 4 || output_size.size() != 2) {
        throw std::runtime_error("aten::adaptive_avg_pool2d on Mini-GPU requires NCHW input and 2D output_size");
    }
    const auto out_h = sym_array_u32(output_size, 0, "adaptive_avg_pool2d output height");
    const auto out_w = sym_array_u32(output_size, 1, "adaptive_avg_pool2d output width");
    if (a.size(2) != static_cast<std::int64_t>(out_h) * 2 ||
        a.size(3) != static_cast<std::int64_t>(out_w) * 2) {
        throw std::runtime_error("aten::adaptive_avg_pool2d on Mini-GPU currently supports exact 2x downsampling only");
    }
    auto out = at::empty({a.size(0), a.size(1), out_h, out_w}, a.options());
    runtime_context().launch_kernel(
        "adaptive_avg_pool2d.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(a)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(out.numel(), "adaptive_avg_pool2d output size")),
            minigpu::KernelArg::u32(checked_u32(a.size(1), "adaptive_avg_pool2d channels")),
            minigpu::KernelArg::u32(checked_u32(a.size(2), "adaptive_avg_pool2d input height")),
            minigpu::KernelArg::u32(checked_u32(a.size(3), "adaptive_avg_pool2d input width")),
            minigpu::KernelArg::u32(out_h),
            minigpu::KernelArg::u32(out_w),
        });
    return out;
}

at::Tensor mse_loss(const at::Tensor &input, const at::Tensor &target) {
    require_fp32_minigpu_contiguous(input, "minigpu::mse_loss input");
    require_fp32_minigpu_contiguous(target, "minigpu::mse_loss target");
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::mse_loss requires matching shapes");
    }
    if (input.numel() <= 0) {
        throw std::runtime_error("minigpu::mse_loss requires non-empty input");
    }
    const auto count = checked_u32(input.numel(), "mse_loss input size");
    auto out = at::empty({}, input.options());
    runtime_context().launch_kernel(
        "mse_loss.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(count),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(count)),
        });
    return out;
}

at::Tensor l1_loss(const at::Tensor &input, const at::Tensor &target) {
    require_fp32_minigpu_contiguous(input, "minigpu::l1_loss input");
    require_fp32_minigpu_contiguous(target, "minigpu::l1_loss target");
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::l1_loss requires matching shapes");
    }
    if (input.numel() <= 0) {
        throw std::runtime_error("minigpu::l1_loss requires non-empty input");
    }
    const auto count = checked_u32(input.numel(), "l1_loss input size");
    auto out = at::empty({}, input.options());
    runtime_context().launch_kernel(
        "l1_loss.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(count),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(count)),
        });
    return out;
}

at::Tensor l2_loss(const at::Tensor &input, const at::Tensor &target) {
    require_fp32_minigpu_contiguous(input, "minigpu::l2_loss input");
    require_fp32_minigpu_contiguous(target, "minigpu::l2_loss target");
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::l2_loss requires matching shapes");
    }
    if (input.numel() <= 0) {
        throw std::runtime_error("minigpu::l2_loss requires non-empty input");
    }
    auto out = at::empty({}, input.options());
    runtime_context().launch_kernel(
        "l2_loss.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(input.numel(), "l2_loss input size")),
        });
    return out;
}

at::Tensor cross_entropy_loss(const at::Tensor &logits, const at::Tensor &target) {
    require_fp32_minigpu_contiguous(logits, "minigpu::cross_entropy logits");
    require_minigpu_contiguous(target, "minigpu::cross_entropy target");
    if (target.scalar_type() != at::kInt) {
        throw std::runtime_error("minigpu::cross_entropy target must be int32 class indices");
    }
    if (logits.dim() != 2) {
        throw std::runtime_error("minigpu::cross_entropy logits must have shape [batch, classes]");
    }
    if (target.dim() != 1 || target.size(0) != logits.size(0)) {
        throw std::runtime_error("minigpu::cross_entropy target must have shape [batch]");
    }
    if (logits.size(0) <= 0 || logits.size(1) <= 0) {
        throw std::runtime_error("minigpu::cross_entropy requires non-empty logits");
    }
    const auto rows = checked_u32(logits.size(0), "cross_entropy batch");
    const auto cols = checked_u32(logits.size(1), "cross_entropy classes");
    auto stats = at::empty({logits.size(0), 3}, logits.options());
    auto out = at::empty({}, logits.options());
    runtime_context().launch_kernel(
        "cross_entropy_stats.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(logits)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(stats)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::u32(cols),
        });
    runtime_context().launch_kernel(
        "cross_entropy_finish.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(stats)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(rows)),
        });
    return out;
}

at::Tensor cross_entropy_debug(const at::Tensor &logits, const at::Tensor &target) {
    require_fp32_minigpu_contiguous(logits, "minigpu::cross_entropy_debug logits");
    require_minigpu_contiguous(target, "minigpu::cross_entropy_debug target");
    if (target.scalar_type() != at::kInt) {
        throw std::runtime_error("minigpu::cross_entropy_debug target must be int32 class indices");
    }
    if (logits.dim() != 2) {
        throw std::runtime_error("minigpu::cross_entropy_debug logits must have shape [batch, classes]");
    }
    if (target.dim() != 1 || target.size(0) != logits.size(0)) {
        throw std::runtime_error("minigpu::cross_entropy_debug target must have shape [batch]");
    }
    if (logits.size(0) <= 0 || logits.size(1) <= 0) {
        throw std::runtime_error("minigpu::cross_entropy_debug requires non-empty logits");
    }
    const auto rows = checked_u32(logits.size(0), "cross_entropy_debug batch");
    const auto cols = checked_u32(logits.size(1), "cross_entropy_debug classes");
    auto stats = at::empty({logits.size(0), 3}, logits.options());
    auto out = at::empty({logits.size(0), 5}, logits.options());
    runtime_context().launch_kernel(
        "cross_entropy_stats.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(logits)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(stats)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::u32(cols),
        });
    runtime_context().launch_kernel(
        "cross_entropy_debug_finish.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(stats)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(rows),
        });
    return out;
}

at::Tensor threshold_backward(
    const at::Tensor &grad_output,
    const at::Tensor &self,
    const at::Scalar &threshold) {
    require_fp32_minigpu_contiguous(grad_output, "aten::threshold_backward grad_output");
    require_fp32_minigpu_contiguous(self, "aten::threshold_backward self");
    if (grad_output.sizes() != self.sizes()) {
        throw std::runtime_error("aten::threshold_backward on Mini-GPU requires matching shapes");
    }
    auto out = at::empty_like(self);
    runtime_context().launch_kernel(
        "relu_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(self)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::f32(threshold.toFloat()),
            minigpu::KernelArg::u32(checked_u32(self.numel(), "threshold_backward size")),
        });
    return out;
}

at::Tensor mse_loss_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &target) {
    require_fp32_minigpu_contiguous(grad_output, "minigpu::mse_loss_backward grad_output");
    require_fp32_minigpu_contiguous(input, "minigpu::mse_loss_backward input");
    require_fp32_minigpu_contiguous(target, "minigpu::mse_loss_backward target");
    if (grad_output.numel() != 1) {
        throw std::runtime_error("minigpu::mse_loss_backward requires scalar grad_output");
    }
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::mse_loss_backward requires matching shapes");
    }
    const auto count = checked_u32(input.numel(), "mse_loss_backward input size");
    auto out = at::empty_like(input);
    runtime_context().launch_kernel(
        "mse_loss_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(count),
            minigpu::KernelArg::f32(2.0f / static_cast<float>(count)),
        });
    return out;
}

at::Tensor l1_loss_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &target) {
    require_fp32_minigpu_contiguous(grad_output, "minigpu::l1_loss_backward grad_output");
    require_fp32_minigpu_contiguous(input, "minigpu::l1_loss_backward input");
    require_fp32_minigpu_contiguous(target, "minigpu::l1_loss_backward target");
    if (grad_output.numel() != 1) {
        throw std::runtime_error("minigpu::l1_loss_backward requires scalar grad_output");
    }
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::l1_loss_backward requires matching shapes");
    }
    const auto count = checked_u32(input.numel(), "l1_loss_backward input size");
    auto out = at::empty_like(input);
    runtime_context().launch_kernel(
        "l1_loss_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(count),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(count)),
        });
    return out;
}

at::Tensor l2_loss_backward(
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &target) {
    require_fp32_minigpu_contiguous(grad_output, "minigpu::l2_loss_backward grad_output");
    require_fp32_minigpu_contiguous(input, "minigpu::l2_loss_backward input");
    require_fp32_minigpu_contiguous(target, "minigpu::l2_loss_backward target");
    if (grad_output.numel() != 1) {
        throw std::runtime_error("minigpu::l2_loss_backward requires scalar grad_output");
    }
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("minigpu::l2_loss_backward requires matching shapes");
    }
    auto out = at::empty_like(input);
    runtime_context().launch_kernel(
        "l2_loss_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(input)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(input.numel(), "l2_loss_backward input size")),
        });
    return out;
}

at::Tensor cross_entropy_loss_backward(
    const at::Tensor &grad_output,
    const at::Tensor &logits,
    const at::Tensor &target) {
    require_fp32_minigpu_contiguous(grad_output, "minigpu::cross_entropy_backward grad_output");
    require_fp32_minigpu_contiguous(logits, "minigpu::cross_entropy_backward logits");
    require_minigpu_contiguous(target, "minigpu::cross_entropy_backward target");
    if (target.scalar_type() != at::kInt) {
        throw std::runtime_error("minigpu::cross_entropy_backward target must be int32 class indices");
    }
    if (grad_output.numel() != 1) {
        throw std::runtime_error("minigpu::cross_entropy_backward requires scalar grad_output");
    }
    if (logits.dim() != 2) {
        throw std::runtime_error("minigpu::cross_entropy_backward logits must have shape [batch, classes]");
    }
    if (target.dim() != 1 || target.size(0) != logits.size(0)) {
        throw std::runtime_error("minigpu::cross_entropy_backward target must have shape [batch]");
    }
    const auto rows = checked_u32(logits.size(0), "cross_entropy_backward batch");
    const auto cols = checked_u32(logits.size(1), "cross_entropy_backward classes");
    auto out = at::empty_like(logits);
    runtime_context().launch_kernel(
        "cross_entropy_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(logits)),
            minigpu::KernelArg::device_ptr(device_address(target)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::u32(cols),
            minigpu::KernelArg::f32(1.0f / static_cast<float>(rows)),
        });
    return out;
}

at::Tensor sigmoid_backward(const at::Tensor &grad_output, const at::Tensor &output) {
    require_fp32_minigpu_contiguous(grad_output, "aten::sigmoid_backward grad_output");
    require_fp32_minigpu_contiguous(output, "aten::sigmoid_backward output");
    if (grad_output.sizes() != output.sizes()) {
        throw std::runtime_error("aten::sigmoid_backward on Mini-GPU requires matching shapes");
    }
    auto out = at::empty_like(output);
    runtime_context().launch_kernel(
        "sigmoid_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(output)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(output.numel(), "sigmoid_backward size")),
        });
    return out;
}

at::Tensor tanh_backward(const at::Tensor &grad_output, const at::Tensor &output) {
    require_fp32_minigpu_contiguous(grad_output, "aten::tanh_backward grad_output");
    require_fp32_minigpu_contiguous(output, "aten::tanh_backward output");
    if (grad_output.sizes() != output.sizes()) {
        throw std::runtime_error("aten::tanh_backward on Mini-GPU requires matching shapes");
    }
    auto out = at::empty_like(output);
    runtime_context().launch_kernel(
        "tanh_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(output)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(checked_u32(output.numel(), "tanh_backward size")),
        });
    return out;
}

at::Tensor softmax_backward_data(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    std::int64_t dim,
    at::ScalarType input_dtype) {
    if (input_dtype != at::kFloat) {
        throw std::runtime_error("aten::_softmax_backward_data on Mini-GPU currently supports fp32 only");
    }
    require_fp32_minigpu_contiguous(grad_output, "aten::_softmax_backward_data grad_output");
    require_fp32_minigpu_contiguous(output, "aten::_softmax_backward_data output");
    if (grad_output.sizes() != output.sizes()) {
        throw std::runtime_error("aten::_softmax_backward_data on Mini-GPU requires matching shapes");
    }
    if (output.dim() == 0) {
        throw std::runtime_error("aten::_softmax_backward_data on Mini-GPU requires at least 1D input");
    }
    if (dim < 0) {
        dim += output.dim();
    }
    if (dim != output.dim() - 1) {
        throw std::runtime_error("aten::_softmax_backward_data on Mini-GPU currently supports the last dimension only");
    }
    auto out = at::empty_like(output);
    const auto cols = checked_u32(output.size(dim), "softmax backward cols");
    const auto rows = checked_u32(output.numel() / output.size(dim), "softmax backward rows");
    runtime_context().launch_kernel(
        "softmax_backward.fp32",
        {
            minigpu::KernelArg::device_ptr(device_address(grad_output)),
            minigpu::KernelArg::device_ptr(device_address(output)),
            minigpu::KernelArg::device_ptr(device_address(out)),
            minigpu::KernelArg::u32(rows),
            minigpu::KernelArg::u32(cols),
        });
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor &self,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    std::array<bool, 3> output_mask) {
    require_fp32_minigpu_contiguous(self, "aten::linear_backward self");
    require_fp32_minigpu_contiguous(grad_output, "aten::linear_backward grad_output");
    require_fp32_minigpu_contiguous(weight, "aten::linear_backward weight");
    if (weight.dim() != 2) {
        throw std::runtime_error("aten::linear_backward on Mini-GPU requires a 2D weight tensor");
    }
    if (self.dim() != 1 && self.dim() != 2) {
        throw std::runtime_error("aten::linear_backward on Mini-GPU currently supports 1D or 2D input");
    }
    const bool input_was_1d = self.dim() == 1;
    const auto batch = checked_u32(input_was_1d ? 1 : self.size(0), "linear_backward batch");
    const auto in_features = checked_u32(input_was_1d ? self.size(0) : self.size(1), "linear_backward in_features");
    const auto out_features = checked_u32(weight.size(0), "linear_backward out_features");
    if (weight.size(1) != static_cast<std::int64_t>(in_features)) {
        throw std::runtime_error("aten::linear_backward on Mini-GPU requires compatible input and weight");
    }
    if (input_was_1d) {
        if (grad_output.dim() != 1 || grad_output.size(0) != static_cast<std::int64_t>(out_features)) {
            throw std::runtime_error("aten::linear_backward on Mini-GPU requires 1D grad_output shape [out_features]");
        }
    } else if (grad_output.dim() != 2 ||
               grad_output.size(0) != static_cast<std::int64_t>(batch) ||
               grad_output.size(1) != static_cast<std::int64_t>(out_features)) {
        throw std::runtime_error("aten::linear_backward on Mini-GPU requires 2D grad_output shape [batch, out_features]");
    }

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0]) {
        grad_input = at::empty_like(self);
        runtime_context().launch_kernel(
            "linear_backward_input.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(weight)),
                minigpu::KernelArg::device_ptr(device_address(grad_input)),
                minigpu::KernelArg::u32(batch),
                minigpu::KernelArg::u32(out_features),
                minigpu::KernelArg::u32(in_features),
            });
    }
    if (output_mask[1]) {
        grad_weight = at::empty_like(weight);
        runtime_context().launch_kernel(
            "linear_backward_weight.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(self)),
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(grad_weight)),
                minigpu::KernelArg::u32(batch),
                minigpu::KernelArg::u32(out_features),
                minigpu::KernelArg::u32(in_features),
            });
    }
    if (output_mask[2]) {
        grad_bias = at::empty({out_features}, grad_output.options());
        runtime_context().launch_kernel(
            "linear_backward_bias.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(grad_bias)),
                minigpu::KernelArg::u32(batch),
                minigpu::KernelArg::u32(out_features),
            });
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor &input,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    std::int64_t groups,
    std::array<bool, 3> output_mask) {
    require_fp32_minigpu_contiguous(input, "aten::convolution_backward input");
    require_fp32_minigpu_contiguous(grad_output, "aten::convolution_backward grad_output");
    require_fp32_minigpu_contiguous(weight, "aten::convolution_backward weight");
    if (transposed) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU does not support transposed convolution yet");
    }
    if (groups != 1) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU currently supports groups=1 only");
    }
    if (input.dim() != 3 && input.dim() != 4) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU currently supports NCL and NCHW input");
    }
    if (weight.dim() != input.dim() || grad_output.dim() != input.dim()) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU requires matching input/weight/grad ranks");
    }
    if (!int_array_all(dilation, 1)) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU currently supports dilation=1 only");
    }
    if (!int_array_all(output_padding, 0)) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU requires output_padding=0");
    }

    const auto batch_size = checked_u32(input.size(0), "conv backward batch_size");
    const auto in_channels = checked_u32(input.size(1), "conv backward in_channels");
    const auto out_channels = checked_u32(weight.size(0), "conv backward out_channels");
    if (weight.size(1) != static_cast<std::int64_t>(in_channels)) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU requires matching input channels");
    }
    if (grad_output.size(0) != input.size(0) || grad_output.size(1) != weight.size(0)) {
        throw std::runtime_error("aten::convolution_backward on Mini-GPU requires compatible grad_output shape");
    }

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (input.dim() == 3) {
        if (stride.size() != 1 || padding.size() != 1 || dilation.size() != 1 ||
            output_padding.size() != 1) {
            throw std::runtime_error("aten::convolution_backward 1D expects one stride/padding/dilation value");
        }
        const auto input_width = checked_u32(input.size(2), "conv backward input_width");
        const auto output_width = checked_u32(grad_output.size(2), "conv backward output_width");
        const auto kernel_width = checked_u32(weight.size(2), "conv backward kernel_width");
        const auto stride_w = checked_u32(stride[0], "conv backward stride");
        const auto padding_w = checked_u32(padding[0], "conv backward padding");
        if (padding_w != 0) {
            throw std::runtime_error("aten::convolution_backward 1D on Mini-GPU currently supports padding=0 only");
        }
        if (stride_w == 0) {
            throw std::runtime_error("aten::convolution_backward 1D on Mini-GPU requires nonzero stride");
        }

        if (output_mask[0]) {
            grad_input = at::empty_like(input);
            runtime_context().launch_kernel(
                "conv1d_backward_input.fp32",
                {
                    minigpu::KernelArg::device_ptr(device_address(grad_output)),
                    minigpu::KernelArg::device_ptr(device_address(weight)),
                    minigpu::KernelArg::device_ptr(device_address(grad_input)),
                    minigpu::KernelArg::u32(checked_u32(input.numel(), "conv1d backward input size")),
                    minigpu::KernelArg::u32(batch_size),
                    minigpu::KernelArg::u32(in_channels),
                    minigpu::KernelArg::u32(out_channels),
                    minigpu::KernelArg::u32(input_width),
                    minigpu::KernelArg::u32(output_width),
                    minigpu::KernelArg::u32(kernel_width),
                    minigpu::KernelArg::u32(stride_w),
                    minigpu::KernelArg::u32(padding_w),
                });
        }
        if (output_mask[1]) {
            grad_weight = at::empty_like(weight);
            runtime_context().launch_kernel(
                "conv1d_backward_weight.fp32",
                {
                    minigpu::KernelArg::device_ptr(device_address(input)),
                    minigpu::KernelArg::device_ptr(device_address(grad_output)),
                    minigpu::KernelArg::device_ptr(device_address(grad_weight)),
                    minigpu::KernelArg::u32(checked_u32(weight.numel(), "conv1d backward weight size")),
                    minigpu::KernelArg::u32(batch_size),
                    minigpu::KernelArg::u32(in_channels),
                    minigpu::KernelArg::u32(out_channels),
                    minigpu::KernelArg::u32(input_width),
                    minigpu::KernelArg::u32(output_width),
                    minigpu::KernelArg::u32(kernel_width),
                    minigpu::KernelArg::u32(stride_w),
                    minigpu::KernelArg::u32(padding_w),
                });
        }
        if (output_mask[2]) {
            grad_bias = at::empty({out_channels}, grad_output.options());
            runtime_context().launch_kernel(
                "conv1d_backward_bias.fp32",
                {
                    minigpu::KernelArg::device_ptr(device_address(grad_output)),
                    minigpu::KernelArg::device_ptr(device_address(grad_bias)),
                    minigpu::KernelArg::u32(batch_size),
                    minigpu::KernelArg::u32(out_channels),
                    minigpu::KernelArg::u32(output_width),
                });
        }
        return std::make_tuple(grad_input, grad_weight, grad_bias);
    }

    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
        output_padding.size() != 2) {
        throw std::runtime_error("aten::convolution_backward 2D expects two stride/padding/dilation values");
    }
    const auto input_height = checked_u32(input.size(2), "conv backward input_height");
    const auto input_width = checked_u32(input.size(3), "conv backward input_width");
    const auto output_height = checked_u32(grad_output.size(2), "conv backward output_height");
    const auto output_width = checked_u32(grad_output.size(3), "conv backward output_width");
    const auto kernel_height = checked_u32(weight.size(2), "conv backward kernel_height");
    const auto kernel_width = checked_u32(weight.size(3), "conv backward kernel_width");
    const auto stride_h = checked_u32(stride[0], "conv backward stride_h");
    const auto stride_w = checked_u32(stride[1], "conv backward stride_w");
    const auto padding_h = checked_u32(padding[0], "conv backward padding_h");
    const auto padding_w = checked_u32(padding[1], "conv backward padding_w");
    if (padding_h != 0 || padding_w != 0) {
        throw std::runtime_error("aten::convolution_backward 2D on Mini-GPU currently supports padding=0 only");
    }
    if (stride_h == 0 || stride_w == 0) {
        throw std::runtime_error("aten::convolution_backward 2D on Mini-GPU requires nonzero stride");
    }

    if (output_mask[0]) {
        grad_input = at::empty_like(input);
        runtime_context().launch_kernel(
            "conv2d_backward_input.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(weight)),
                minigpu::KernelArg::device_ptr(device_address(grad_input)),
                minigpu::KernelArg::u32(checked_u32(input.numel(), "conv2d backward input size")),
                minigpu::KernelArg::u32(batch_size),
                minigpu::KernelArg::u32(in_channels),
                minigpu::KernelArg::u32(out_channels),
                minigpu::KernelArg::u32(input_height),
                minigpu::KernelArg::u32(input_width),
                minigpu::KernelArg::u32(output_height),
                minigpu::KernelArg::u32(output_width),
                minigpu::KernelArg::u32(kernel_height),
                minigpu::KernelArg::u32(kernel_width),
                minigpu::KernelArg::u32(stride_h),
                minigpu::KernelArg::u32(stride_w),
                minigpu::KernelArg::u32(padding_h),
                minigpu::KernelArg::u32(padding_w),
            });
    }
    if (output_mask[1]) {
        grad_weight = at::empty_like(weight);
        runtime_context().launch_kernel(
            "conv2d_backward_weight.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(input)),
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(grad_weight)),
                minigpu::KernelArg::u32(checked_u32(weight.numel(), "conv2d backward weight size")),
                minigpu::KernelArg::u32(batch_size),
                minigpu::KernelArg::u32(in_channels),
                minigpu::KernelArg::u32(out_channels),
                minigpu::KernelArg::u32(input_height),
                minigpu::KernelArg::u32(input_width),
                minigpu::KernelArg::u32(output_height),
                minigpu::KernelArg::u32(output_width),
                minigpu::KernelArg::u32(kernel_height),
                minigpu::KernelArg::u32(kernel_width),
                minigpu::KernelArg::u32(stride_h),
                minigpu::KernelArg::u32(stride_w),
                minigpu::KernelArg::u32(padding_h),
                minigpu::KernelArg::u32(padding_w),
            });
    }
    if (output_mask[2]) {
        grad_bias = at::empty({out_channels}, grad_output.options());
        runtime_context().launch_kernel(
            "conv2d_backward_bias.fp32",
            {
                minigpu::KernelArg::device_ptr(device_address(grad_output)),
                minigpu::KernelArg::device_ptr(device_address(grad_bias)),
                minigpu::KernelArg::u32(batch_size),
                minigpu::KernelArg::u32(out_channels),
                minigpu::KernelArg::u32(output_height),
                minigpu::KernelArg::u32(output_width),
            });
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace minigpu::torch_backend
