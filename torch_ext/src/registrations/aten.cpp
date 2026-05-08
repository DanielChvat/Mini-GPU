#include "minigpu_torch.hpp"

#include <torch/library.h>

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(minigpu::torch_backend::empty));
    m.impl("empty_strided", TORCH_FN(minigpu::torch_backend::empty_strided));
    m.impl("copy_", TORCH_FN(minigpu::torch_backend::copy_));
    m.impl("view", TORCH_FN(minigpu::torch_backend::view));
    m.impl("as_strided", TORCH_FN(minigpu::torch_backend::as_strided));
    m.impl("_reshape_alias", TORCH_FN(minigpu::torch_backend::reshape_alias));
    m.impl("add.Tensor", TORCH_FN(minigpu::torch_backend::add_tensor));
    m.impl("mul.Tensor", TORCH_FN(minigpu::torch_backend::mul_tensor));
    m.impl("mm", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
    m.impl("exp", TORCH_FN(minigpu::torch_backend::exp));
    m.impl("log", TORCH_FN(minigpu::torch_backend::log));
    m.impl("log2", TORCH_FN(minigpu::torch_backend::log2));
    m.impl("sqrt", TORCH_FN(minigpu::torch_backend::sqrt));
    m.impl("reciprocal", TORCH_FN(minigpu::torch_backend::reciprocal));
    m.impl("pow.Tensor_Tensor", TORCH_FN(minigpu::torch_backend::pow_tensor_tensor));
    m.impl("pow.Tensor_Scalar", TORCH_FN(minigpu::torch_backend::pow_tensor_scalar));
}
