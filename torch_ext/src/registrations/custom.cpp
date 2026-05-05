#include "minigpu_torch.hpp"

#include <torch/library.h>

TORCH_LIBRARY(minigpu, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.def("matmul(Tensor a, Tensor b) -> Tensor");
    m.def("relu(Tensor a) -> Tensor");
    m.def("exp(Tensor a) -> Tensor");
    m.def("log(Tensor a) -> Tensor");
    m.def("log2(Tensor a) -> Tensor");
    m.def("sqrt(Tensor a) -> Tensor");
    m.def("reciprocal(Tensor a) -> Tensor");
    m.def("pow(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(minigpu, PrivateUse1, m) {
    m.impl("vector_add", TORCH_FN(minigpu::torch_backend::vector_add));
    m.impl("matmul", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
    m.impl("exp", TORCH_FN(minigpu::torch_backend::exp));
    m.impl("log", TORCH_FN(minigpu::torch_backend::log));
    m.impl("log2", TORCH_FN(minigpu::torch_backend::log2));
    m.impl("sqrt", TORCH_FN(minigpu::torch_backend::sqrt));
    m.impl("reciprocal", TORCH_FN(minigpu::torch_backend::reciprocal));
    m.impl("pow", TORCH_FN(minigpu::torch_backend::pow_tensor_tensor));
}
