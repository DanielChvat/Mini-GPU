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

} // namespace minigpu::torch_backend
