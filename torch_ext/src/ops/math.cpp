#include "minigpu_torch.hpp"

#include "elementwise.hpp"
#include "minigpu_kernels.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
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

std::string_view supported_kernel_dtype(const at::Tensor &tensor, const char *op_name) {
    if (tensor.scalar_type() == at::kInt) {
        return "i32";
    }
    if (tensor.scalar_type() == at::kFloat) {
        return "fp32";
    }
    throw std::runtime_error(std::string(op_name) + " on Mini-GPU currently supports int32 and fp32");
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
    if (alpha.toDouble() != 1.0) {
        throw std::runtime_error("aten::add.Tensor on Mini-GPU currently requires alpha=1");
    }

    return detail::run_vector_add_kernel(a, b, "aten::add.Tensor");
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
    return to_minigpu_tensor(at::log(to_cpu_tensor(a, "aten::log")), a.options());
}

at::Tensor log2(const at::Tensor &a) {
    return to_minigpu_tensor(at::log2(to_cpu_tensor(a, "aten::log2")), a.options());
}

at::Tensor log10(const at::Tensor &a) {
    return to_minigpu_tensor(at::log10(to_cpu_tensor(a, "aten::log10")), a.options());
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

} // namespace minigpu::torch_backend
