#include "minigpu_torch.hpp"

#include <ATen/core/LegacyTypeDispatch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace {

struct MiniGpuReluAutograd : public torch::autograd::Function<MiniGpuReluAutograd> {
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input) {
        ctx->save_for_backward({input});
        at::AutoDispatchBelowAutograd guard;
        return minigpu::torch_backend::relu(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::AutoDispatchBelowAutograd guard;
        return {minigpu::torch_backend::threshold_backward(grad_outputs[0], saved[0], 0.0)};
    }
};

struct MiniGpuSigmoidAutograd : public torch::autograd::Function<MiniGpuSigmoidAutograd> {
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input) {
        at::AutoDispatchBelowAutograd guard;
        auto output = minigpu::torch_backend::sigmoid(input);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::AutoDispatchBelowAutograd guard;
        return {minigpu::torch_backend::sigmoid_backward(grad_outputs[0], saved[0])};
    }
};

struct MiniGpuTanhAutograd : public torch::autograd::Function<MiniGpuTanhAutograd> {
    static at::Tensor forward(torch::autograd::AutogradContext *ctx, const at::Tensor &input) {
        at::AutoDispatchBelowAutograd guard;
        auto output = minigpu::torch_backend::tanh(input);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        at::AutoDispatchBelowAutograd guard;
        return {minigpu::torch_backend::tanh_backward(grad_outputs[0], saved[0])};
    }
};

struct MiniGpuSoftmaxAutograd : public torch::autograd::Function<MiniGpuSoftmaxAutograd> {
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input,
        std::int64_t dim,
        bool half_to_float) {
        at::AutoDispatchBelowAutograd guard;
        auto output = minigpu::torch_backend::softmax_impl(input, dim, half_to_float);
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["dtype"] = static_cast<std::int64_t>(input.scalar_type());
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const auto dim = ctx->saved_data["dim"].toInt();
        const auto dtype = static_cast<at::ScalarType>(ctx->saved_data["dtype"].toInt());
        at::AutoDispatchBelowAutograd guard;
        return {
            minigpu::torch_backend::softmax_backward_data(grad_outputs[0], saved[0], dim, dtype),
            at::Tensor(),
            at::Tensor(),
        };
    }
};

struct MiniGpuLinearAutograd : public torch::autograd::Function<MiniGpuLinearAutograd> {
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &input,
        const at::Tensor &weight,
        const std::optional<at::Tensor> &bias) {
        ctx->saved_data["has_bias"] = bias.has_value();
        ctx->save_for_backward({input, weight});
        at::AutoDispatchBelowAutograd guard;
        return minigpu::torch_backend::linear(input, weight, bias);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const bool has_bias = ctx->saved_data["has_bias"].toBool();
        at::AutoDispatchBelowAutograd guard;
        auto grads = minigpu::torch_backend::linear_backward(
            saved[0],
            grad_outputs[0],
            saved[1],
            {
                ctx->needs_input_grad(0),
                ctx->needs_input_grad(1),
                has_bias && ctx->needs_input_grad(2),
            });
        return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
    }
};

at::Tensor minigpu_autograd_relu(const at::Tensor &input) {
    return MiniGpuReluAutograd::apply(input);
}

at::Tensor minigpu_autograd_sigmoid(const at::Tensor &input) {
    return MiniGpuSigmoidAutograd::apply(input);
}

at::Tensor minigpu_autograd_tanh(const at::Tensor &input) {
    return MiniGpuTanhAutograd::apply(input);
}

at::Tensor minigpu_autograd_softmax_impl(const at::Tensor &input, std::int64_t dim, bool half_to_float) {
    return MiniGpuSoftmaxAutograd::apply(input, dim, half_to_float);
}

at::Tensor minigpu_autograd_softmax(
    const at::Tensor &input,
    std::int64_t dim,
    std::optional<at::ScalarType> dtype) {
    if (dtype.has_value() && *dtype != input.scalar_type()) {
        return MiniGpuSoftmaxAutograd::apply(input.to(*dtype), dim, false);
    }
    return MiniGpuSoftmaxAutograd::apply(input, dim, false);
}

at::Tensor minigpu_autograd_linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const std::optional<at::Tensor> &bias) {
    return MiniGpuLinearAutograd::apply(input, weight, bias);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(minigpu::torch_backend::empty));
    m.impl("empty_strided", TORCH_FN(minigpu::torch_backend::empty_strided));
    m.impl("copy_", TORCH_FN(minigpu::torch_backend::copy_));
    m.impl("_to_copy", TORCH_FN(minigpu::torch_backend::to_copy));
    m.impl("view", TORCH_FN(minigpu::torch_backend::view));
    m.impl("as_strided", TORCH_FN(minigpu::torch_backend::as_strided));
    m.impl("_reshape_alias", TORCH_FN(minigpu::torch_backend::reshape_alias));
    m.impl("add.Tensor", TORCH_FN(minigpu::torch_backend::add_tensor));
    m.impl("div.Tensor", TORCH_FN(minigpu::torch_backend::div_tensor));
    m.impl("mul.Tensor", TORCH_FN(minigpu::torch_backend::mul_tensor));
    m.impl("mm", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("mv", TORCH_FN(minigpu::torch_backend::mv));
    m.impl("dot", TORCH_FN(minigpu::torch_backend::dot));
    m.impl("addmm", TORCH_FN(minigpu::torch_backend::addmm));
    m.impl("linear", TORCH_FN(minigpu::torch_backend::linear));
    m.impl("convolution", TORCH_FN(minigpu::torch_backend::convolution));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
    m.impl("exp", TORCH_FN(minigpu::torch_backend::exp));
    m.impl("log", TORCH_FN(minigpu::torch_backend::log));
    m.impl("log2", TORCH_FN(minigpu::torch_backend::log2));
    m.impl("log10", TORCH_FN(minigpu::torch_backend::log10));
    m.impl("sqrt", TORCH_FN(minigpu::torch_backend::sqrt));
    m.impl("reciprocal", TORCH_FN(minigpu::torch_backend::reciprocal));
    m.impl("pow.Tensor_Tensor", TORCH_FN(minigpu::torch_backend::pow_tensor_tensor));
    m.impl("pow.Tensor_Scalar", TORCH_FN(minigpu::torch_backend::pow_tensor_scalar));
    m.impl("sigmoid", TORCH_FN(minigpu::torch_backend::sigmoid));
    m.impl("tanh", TORCH_FN(minigpu::torch_backend::tanh));
    m.impl("sin", TORCH_FN(minigpu::torch_backend::sin));
    m.impl("cos", TORCH_FN(minigpu::torch_backend::cos));
    m.impl("tan", TORCH_FN(minigpu::torch_backend::tan));
    m.impl("sum", TORCH_FN(minigpu::torch_backend::sum));
    m.impl("sum.dim_IntList", TORCH_FN(minigpu::torch_backend::sum_dim));
    m.impl("mean", TORCH_FN(minigpu::torch_backend::mean));
    m.impl("mean.dim", TORCH_FN(minigpu::torch_backend::mean_dim));
    m.impl("amax", TORCH_FN(minigpu::torch_backend::amax));
    m.impl("amin", TORCH_FN(minigpu::torch_backend::amin));
    m.impl("argmax", TORCH_FN(minigpu::torch_backend::argmax));
    m.impl("argmin", TORCH_FN(minigpu::torch_backend::argmin));
    m.impl("softmax.int", TORCH_FN(minigpu::torch_backend::softmax));
    m.impl("_softmax", TORCH_FN(minigpu::torch_backend::softmax_impl));
    m.impl("threshold_backward", TORCH_FN(minigpu::torch_backend::threshold_backward));
    m.impl("sigmoid_backward", TORCH_FN(minigpu::torch_backend::sigmoid_backward));
    m.impl("tanh_backward", TORCH_FN(minigpu::torch_backend::tanh_backward));
    m.impl("_softmax_backward_data", TORCH_FN(minigpu::torch_backend::softmax_backward_data));
    m.impl("linear_backward", TORCH_FN(minigpu::torch_backend::linear_backward));
    m.impl("max_pool2d", TORCH_FN(minigpu::torch_backend::max_pool2d));
    m.impl("avg_pool2d", TORCH_FN(minigpu::torch_backend::avg_pool2d));
    m.impl("adaptive_avg_pool2d", TORCH_FN(minigpu::torch_backend::adaptive_avg_pool2d));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("relu", TORCH_FN(minigpu_autograd_relu));
    m.impl("sigmoid", TORCH_FN(minigpu_autograd_sigmoid));
    m.impl("tanh", TORCH_FN(minigpu_autograd_tanh));
    m.impl("_softmax", TORCH_FN(minigpu_autograd_softmax_impl));
    m.impl("softmax.int", TORCH_FN(minigpu_autograd_softmax));
    m.impl("linear", TORCH_FN(minigpu_autograd_linear));
}
