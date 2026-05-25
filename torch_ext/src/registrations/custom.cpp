#include "minigpu_torch.hpp"

#include <ATen/core/LegacyTypeDispatch.h>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace {

enum class MiniGpuLossKind {
    Mse,
    L1,
    L2,
    CrossEntropy,
};

at::Tensor run_loss_forward(
    MiniGpuLossKind kind,
    const at::Tensor &input,
    const at::Tensor &target) {
    switch (kind) {
    case MiniGpuLossKind::Mse:
        return minigpu::torch_backend::mse_loss(input, target);
    case MiniGpuLossKind::L1:
        return minigpu::torch_backend::l1_loss(input, target);
    case MiniGpuLossKind::L2:
        return minigpu::torch_backend::l2_loss(input, target);
    case MiniGpuLossKind::CrossEntropy:
        return minigpu::torch_backend::cross_entropy_loss(input, target);
    }
    throw std::runtime_error("unknown Mini-GPU loss kind");
}

at::Tensor run_loss_backward(
    MiniGpuLossKind kind,
    const at::Tensor &grad_output,
    const at::Tensor &input,
    const at::Tensor &target) {
    switch (kind) {
    case MiniGpuLossKind::Mse:
        return minigpu::torch_backend::mse_loss_backward(grad_output, input, target);
    case MiniGpuLossKind::L1:
        return minigpu::torch_backend::l1_loss_backward(grad_output, input, target);
    case MiniGpuLossKind::L2:
        return minigpu::torch_backend::l2_loss_backward(grad_output, input, target);
    case MiniGpuLossKind::CrossEntropy:
        return minigpu::torch_backend::cross_entropy_loss_backward(grad_output, input, target);
    }
    throw std::runtime_error("unknown Mini-GPU loss kind");
}

struct MiniGpuLossAutograd : public torch::autograd::Function<MiniGpuLossAutograd> {
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input,
        const at::Tensor &target,
        std::int64_t kind_value) {
        ctx->saved_data["kind"] = kind_value;
        ctx->save_for_backward({input, target});
        at::AutoDispatchBelowAutograd guard;
        return run_loss_forward(static_cast<MiniGpuLossKind>(kind_value), input, target);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const auto kind = static_cast<MiniGpuLossKind>(ctx->saved_data["kind"].toInt());
        at::AutoDispatchBelowAutograd guard;
        at::Tensor grad_input;
        at::Tensor grad_target;
        if (ctx->needs_input_grad(0) || ctx->needs_input_grad(1)) {
            grad_input = run_loss_backward(kind, grad_outputs[0], saved[0], saved[1]);
        }
        if (ctx->needs_input_grad(1) && kind != MiniGpuLossKind::CrossEntropy) {
            grad_target = minigpu::torch_backend::scal(grad_input, -1.0);
        }
        return {ctx->needs_input_grad(0) ? grad_input : at::Tensor(), grad_target, at::Tensor()};
    }
};

at::Tensor minigpu_autograd_mse_loss(const at::Tensor &input, const at::Tensor &target) {
    return MiniGpuLossAutograd::apply(input, target, static_cast<std::int64_t>(MiniGpuLossKind::Mse));
}

at::Tensor minigpu_autograd_l1_loss(const at::Tensor &input, const at::Tensor &target) {
    return MiniGpuLossAutograd::apply(input, target, static_cast<std::int64_t>(MiniGpuLossKind::L1));
}

at::Tensor minigpu_autograd_l2_loss(const at::Tensor &input, const at::Tensor &target) {
    return MiniGpuLossAutograd::apply(input, target, static_cast<std::int64_t>(MiniGpuLossKind::L2));
}

at::Tensor minigpu_autograd_cross_entropy(const at::Tensor &input, const at::Tensor &target) {
    return MiniGpuLossAutograd::apply(input, target, static_cast<std::int64_t>(MiniGpuLossKind::CrossEntropy));
}

} // namespace

TORCH_LIBRARY(minigpu, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.def("matmul(Tensor a, Tensor b) -> Tensor");
    m.def("mv(Tensor a, Tensor x) -> Tensor");
    m.def("dot(Tensor a, Tensor b) -> Tensor");
    m.def("scal(Tensor x, Scalar alpha) -> Tensor");
    m.def("axpy(Tensor x, Tensor y, Scalar alpha) -> Tensor");
    m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1) -> Tensor");
    m.def("linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
    m.def("relu(Tensor a) -> Tensor");
    m.def("exp(Tensor a) -> Tensor");
    m.def("log(Tensor a) -> Tensor");
    m.def("log2(Tensor a) -> Tensor");
    m.def("log10(Tensor a) -> Tensor");
    m.def("sqrt(Tensor a) -> Tensor");
    m.def("reciprocal(Tensor a) -> Tensor");
    m.def("div(Tensor a, Tensor b) -> Tensor");
    m.def("pow(Tensor a, Tensor b) -> Tensor");
    m.def("sigmoid(Tensor a) -> Tensor");
    m.def("tanh(Tensor a) -> Tensor");
    m.def("sin(Tensor a) -> Tensor");
    m.def("cos(Tensor a) -> Tensor");
    m.def("tan(Tensor a) -> Tensor");
    m.def("sum(Tensor a) -> Tensor");
    m.def("mean(Tensor a) -> Tensor");
    m.def("amax(Tensor a) -> Tensor");
    m.def("amin(Tensor a) -> Tensor");
    m.def("argmax(Tensor a) -> Tensor");
    m.def("argmin(Tensor a) -> Tensor");
    m.def("softmax(Tensor a, int dim) -> Tensor");
    m.def("max_pool2d(Tensor a, int[] kernel_size, int[] stride=[]) -> Tensor");
    m.def("avg_pool2d(Tensor a, int[] kernel_size, int[] stride=[]) -> Tensor");
    m.def("adaptive_avg_pool2d(Tensor a, SymInt[] output_size) -> Tensor");
    m.def("mse_loss(Tensor input, Tensor target) -> Tensor");
    m.def("l1_loss(Tensor input, Tensor target) -> Tensor");
    m.def("l2_loss(Tensor input, Tensor target) -> Tensor");
    m.def("cross_entropy(Tensor input, Tensor target) -> Tensor");
    m.def("cross_entropy_debug(Tensor input, Tensor target) -> Tensor");
}

TORCH_LIBRARY_IMPL(minigpu, PrivateUse1, m) {
    m.impl("vector_add", TORCH_FN(minigpu::torch_backend::vector_add));
    m.impl("matmul", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("mv", TORCH_FN(minigpu::torch_backend::mv));
    m.impl("dot", TORCH_FN(minigpu::torch_backend::dot));
    m.impl("scal", TORCH_FN(minigpu::torch_backend::scal));
    m.impl("axpy", TORCH_FN(minigpu::torch_backend::axpy));
    m.impl("addmm", TORCH_FN(minigpu::torch_backend::addmm));
    m.impl("linear", TORCH_FN(minigpu::torch_backend::linear));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
    m.impl("exp", TORCH_FN(minigpu::torch_backend::exp));
    m.impl("log", TORCH_FN(minigpu::torch_backend::log));
    m.impl("log2", TORCH_FN(minigpu::torch_backend::log2));
    m.impl("log10", TORCH_FN(minigpu::torch_backend::log10));
    m.impl("sqrt", TORCH_FN(minigpu::torch_backend::sqrt));
    m.impl("reciprocal", TORCH_FN(minigpu::torch_backend::reciprocal));
    m.impl("div", TORCH_FN(minigpu::torch_backend::div_tensor));
    m.impl("pow", TORCH_FN(minigpu::torch_backend::pow_tensor_tensor));
    m.impl("sigmoid", TORCH_FN(minigpu::torch_backend::sigmoid));
    m.impl("tanh", TORCH_FN(minigpu::torch_backend::tanh));
    m.impl("sin", TORCH_FN(minigpu::torch_backend::sin));
    m.impl("cos", TORCH_FN(minigpu::torch_backend::cos));
    m.impl("tan", TORCH_FN(minigpu::torch_backend::tan));
    m.impl("sum", [](const at::Tensor &a) {
        return minigpu::torch_backend::sum(a, c10::nullopt);
    });
    m.impl("mean", [](const at::Tensor &a) {
        return minigpu::torch_backend::mean(a, c10::nullopt);
    });
    m.impl("amax", [](const at::Tensor &a) {
        return minigpu::torch_backend::amax(a, {}, false);
    });
    m.impl("amin", [](const at::Tensor &a) {
        return minigpu::torch_backend::amin(a, {}, false);
    });
    m.impl("argmax", [](const at::Tensor &a) {
        return minigpu::torch_backend::argmax(a, std::nullopt, false);
    });
    m.impl("argmin", [](const at::Tensor &a) {
        return minigpu::torch_backend::argmin(a, std::nullopt, false);
    });
    m.impl("softmax", [](const at::Tensor &a, std::int64_t dim) {
        return minigpu::torch_backend::softmax(a, dim, c10::nullopt);
    });
    m.impl("max_pool2d", [](const at::Tensor &a, at::IntArrayRef kernel_size, at::IntArrayRef stride) {
        return minigpu::torch_backend::max_pool2d(a, kernel_size, stride, {0, 0}, {1, 1}, false);
    });
    m.impl("avg_pool2d", [](const at::Tensor &a, at::IntArrayRef kernel_size, at::IntArrayRef stride) {
        return minigpu::torch_backend::avg_pool2d(a, kernel_size, stride, {0, 0}, false, true, std::nullopt);
    });
    m.impl("adaptive_avg_pool2d", TORCH_FN(minigpu::torch_backend::adaptive_avg_pool2d));
    m.impl("mse_loss", TORCH_FN(minigpu::torch_backend::mse_loss));
    m.impl("l1_loss", TORCH_FN(minigpu::torch_backend::l1_loss));
    m.impl("l2_loss", TORCH_FN(minigpu::torch_backend::l2_loss));
    m.impl("cross_entropy", TORCH_FN(minigpu::torch_backend::cross_entropy_loss));
    m.impl("cross_entropy_debug", TORCH_FN(minigpu::torch_backend::cross_entropy_debug));
}

TORCH_LIBRARY_IMPL(minigpu, AutogradPrivateUse1, m) {
    m.impl("mse_loss", TORCH_FN(minigpu_autograd_mse_loss));
    m.impl("l1_loss", TORCH_FN(minigpu_autograd_l1_loss));
    m.impl("l2_loss", TORCH_FN(minigpu_autograd_l2_loss));
    m.impl("cross_entropy", TORCH_FN(minigpu_autograd_cross_entropy));
}
