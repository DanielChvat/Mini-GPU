#include "minigpu_torch.hpp"

#include <torch/library.h>

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(minigpu::torch_backend::empty));
    m.impl("copy_", TORCH_FN(minigpu::torch_backend::copy_));
    m.impl("add.Tensor", TORCH_FN(minigpu::torch_backend::add_tensor));
    m.impl("mul.Tensor", TORCH_FN(minigpu::torch_backend::mul_tensor));
    m.impl("mm", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
}
