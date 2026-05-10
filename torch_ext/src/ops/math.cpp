#include "minigpu_torch.hpp"

#include "elementwise.hpp"
#include "minigpu_kernels.hpp"

#include <cstdint>
#include <stdexcept>

namespace minigpu::torch_backend {

namespace {

/* Throw a consistent message for operators that are intentionally skeletons. */
[[noreturn]] void unimplemented_op(const char *name) {
    throw std::runtime_error(
        std::string("Mini-GPU PyTorch op is a stub: ") + name);
}

} // namespace

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
    (void)a;
    (void)b;

    unimplemented_op("aten::mul.Tensor");
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

at::Tensor relu(const at::Tensor &a) {
    return detail::run_relu_kernel(a, "aten::relu");
}

at::Tensor exp(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::exp");
}

at::Tensor log(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::log");
}

at::Tensor log2(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::log2");
}

at::Tensor sqrt(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::sqrt");
}

at::Tensor reciprocal(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::reciprocal");
}

at::Tensor pow_tensor_tensor(const at::Tensor &a, const at::Tensor &b) {
    (void)a;
    (void)b;

    unimplemented_op("aten::pow.Tensor_Tensor");
}

at::Tensor pow_tensor_scalar(const at::Tensor &a, const at::Scalar &b) {
    (void)a;
    (void)b;

    unimplemented_op("aten::pow.Tensor_Scalar");
}

} // namespace minigpu::torch_backend
