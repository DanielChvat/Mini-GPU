#include "minigpu_torch.hpp"

#include <torch/library.h>

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
}
