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

/* Runtime layout metadata for one vector-add dtype. */
struct DTypeLayout {
    std::string_view name;
};

/* Map PyTorch dtypes to runtime kernel names and packed-word layout. */
DTypeLayout vector_add_dtype(const at::Tensor &tensor, const char *op_name) {
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
        return {"fp8"};
    }

    throw std::runtime_error(
        std::string(op_name) + " does not have a registered Mini-GPU vector-add dtype");
}

} // namespace

at::Tensor run_vector_add_kernel(
    const at::Tensor &a,
    const at::Tensor &b,
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
    const DTypeLayout layout = vector_add_dtype(a, op_name);

    auto out = at::empty_like(a);
    constexpr std::size_t kLanesPerLaunch = 4;

    const auto a_addr = device_address(a);
    const auto b_addr = device_address(b);
    const auto out_addr = device_address(out);
    const auto elements = static_cast<std::size_t>(a.numel());

    for (std::size_t offset = 0; offset < elements; offset += kLanesPerLaunch) {
        const auto lanes = std::min(kLanesPerLaunch, elements - offset);
        const auto byte_offset =
            static_cast<minigpu::DeviceAddress>(offset * a.element_size());
        minigpu::LaunchConfig launch;
        launch.grid_dim = 1;
        launch.block_dim = static_cast<std::uint32_t>(kLanesPerLaunch);
        launch.active_mask = (1u << lanes) - 1u;
        launch.timeout_ms = 5000;

        minigpu::kernels::launch_elementwise_binary(
            runtime_context(),
            "vector_add",
            minigpu::kernels::TensorView{a_addr + byte_offset, lanes, layout.name},
            minigpu::kernels::TensorView{b_addr + byte_offset, lanes, layout.name},
            minigpu::kernels::TensorView{out_addr + byte_offset, lanes, layout.name},
            &launch);
    }

    return out;
}

} // namespace minigpu::torch_backend::detail
