#include "elementwise.hpp"

#include "minigpu_kernels.hpp"
#include "minigpu_torch.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace minigpu::torch_backend::detail {

namespace {

/* Runtime layout metadata for one Mini-GPU kernel dtype. */
struct DTypeLayout {
    std::string_view name;
};

/* Map PyTorch dtypes to runtime kernel names. */
DTypeLayout kernel_dtype(const at::Tensor &tensor, const char *op_name) {
    if (tensor.scalar_type() == at::kInt) {
        return {"i32"};
    }
    if (tensor.scalar_type() == at::kFloat) {
        return {"fp32"};
    }
    if (tensor.scalar_type() == at::kShort) {
        return {"i16"};
    }
    if (tensor.scalar_type() == at::kChar) {
        return {"i8"};
    }
    if (tensor.scalar_type() == at::kHalf) {
        return {"fp16"};
    }
    if (tensor.scalar_type() == c10::ScalarType::Float8_e4m3fn) {
        return {"fp8_e4m3fn"};
    }

    throw std::runtime_error(
        std::string(op_name) + " does not have a registered Mini-GPU dtype");
}

} // namespace

at::Tensor run_vector_add_kernel(
    const at::Tensor &a,
    const at::Tensor &b,
    const char *op_name) {
    return run_binary_kernel(a, b, "vector_add", op_name);
}

at::Tensor run_binary_kernel(
    const at::Tensor &a,
    const at::Tensor &b,
    const char *kernel_op,
    const char *op_name) {
    if (a.device().type() != c10::DeviceType::PrivateUse1 ||
        b.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error(std::string(op_name) + " requires Mini-GPU tensors");
    }
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error(std::string(op_name) + " requires matching shapes");
    }
    if (!a.is_contiguous() || !b.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " requires contiguous tensors");
    }
    if (a.scalar_type() != b.scalar_type()) {
        throw std::runtime_error(std::string(op_name) + " requires matching dtypes");
    }
    const DTypeLayout layout = kernel_dtype(a, op_name);

    auto out = at::empty_like(a);
    const auto a_addr = device_address(a);
    const auto b_addr = device_address(b);
    const auto out_addr = device_address(out);
    const auto elements = static_cast<std::size_t>(a.numel());

    minigpu::kernels::launch_elementwise_binary(
        runtime_context(),
        kernel_op,
        minigpu::kernels::TensorView{a_addr, elements, layout.name},
        minigpu::kernels::TensorView{b_addr, elements, layout.name},
        minigpu::kernels::TensorView{out_addr, elements, layout.name});

    return out;
}

at::Tensor run_unary_kernel(
    const at::Tensor &a,
    const char *kernel_op,
    const char *op_name) {
    if (a.device().type() != c10::DeviceType::PrivateUse1) {
        throw std::runtime_error(std::string(op_name) + " requires a Mini-GPU tensor");
    }
    if (!a.is_contiguous()) {
        throw std::runtime_error(std::string(op_name) + " requires a contiguous tensor");
    }
    const DTypeLayout layout = kernel_dtype(a, op_name);

    auto out = at::empty_like(a);
    const auto a_addr = device_address(a);
    const auto out_addr = device_address(out);
    const auto elements = static_cast<std::size_t>(a.numel());

    minigpu::kernels::launch_elementwise_unary(
        runtime_context(),
        kernel_op,
        minigpu::kernels::TensorView{a_addr, elements, layout.name},
        minigpu::kernels::TensorView{out_addr, elements, layout.name});

    return out;
}

at::Tensor run_relu_kernel(const at::Tensor &a, const char *op_name) {
    return run_unary_kernel(a, "relu", op_name);
}

} // namespace minigpu::torch_backend::detail
