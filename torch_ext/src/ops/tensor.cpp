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

at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
    (void)size;
    (void)dtype;
    (void)layout;
    (void)device;
    (void)pin_memory;
    (void)memory_format;

    (void)runtime_context;
    unimplemented_op("aten::empty.memory_format");
}

at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking) {
    (void)self;
    (void)src;
    (void)non_blocking;

    unimplemented_op("aten::copy_");
}

} // namespace minigpu::torch_backend
