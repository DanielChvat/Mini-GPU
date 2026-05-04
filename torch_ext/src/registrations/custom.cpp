#include "minigpu_torch.hpp"

#include <torch/library.h>

TORCH_LIBRARY(minigpu, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.def("matmul(Tensor a, Tensor b) -> Tensor");
    m.def("relu(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(minigpu, PrivateUse1, m) {
    m.impl("vector_add", TORCH_FN(minigpu::torch_backend::vector_add));
    m.impl("matmul", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
}
