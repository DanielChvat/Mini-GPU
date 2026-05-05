#include "minigpu_torch.hpp"

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
    (void)a;
    (void)b;
    (void)alpha;

    unimplemented_op("aten::add.Tensor");
}

at::Tensor vector_add(const at::Tensor &a, const at::Tensor &b) {
    (void)a;
    (void)b;

    unimplemented_op("minigpu::vector_add");
}

at::Tensor mul_tensor(const at::Tensor &a, const at::Tensor &b) {
    (void)a;
    (void)b;

    unimplemented_op("aten::mul.Tensor");
}

at::Tensor mm(const at::Tensor &a, const at::Tensor &b) {
    (void)a;
    (void)b;

    unimplemented_op("aten::mm");
}

at::Tensor relu(const at::Tensor &a) {
    (void)a;

    unimplemented_op("aten::relu");
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
