#include "minigpu_torch.hpp"

#include "elementwise.hpp"
#include "minigpu_kernels.hpp"

#include <cstdint>
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

at::Tensor linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const ::std::optional<at::Tensor> &bias) {
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

    std::string_view dtype;
    if (input.scalar_type() == at::kInt) {
        dtype = "i32";
    } else if (input.scalar_type() == at::kFloat) {
        dtype = "fp32";
    } else {
        throw std::runtime_error("aten::linear on Mini-GPU currently supports int32 and fp32");
    }

    if (bias.has_value() && bias->defined()) {
        throw std::runtime_error("aten::linear on Mini-GPU bias is not implemented yet");
    }

    at::Tensor input_2d = input.dim() == 1 ? input.view({1, input.size(0)}) : input;
    auto out_2d = at::empty({input_2d.size(0), weight.size(0)}, input.options());
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
    return input.dim() == 1 ? out_2d.view({weight.size(0)}) : out_2d;
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
