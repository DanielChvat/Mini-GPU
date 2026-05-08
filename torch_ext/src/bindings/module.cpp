#include "minigpu_torch.hpp"

#include <torch/extension.h>

PYBIND11_MODULE(minigpu_torch, m) {
    m.doc() = "Mini-GPU PyTorch PrivateUse1 backend stubs";
    m.def("init", &minigpu::torch_backend::init);
    m.def("is_built", &minigpu::torch_backend::is_built);
    m.def("is_available", &minigpu::torch_backend::is_available);
    m.def("device_count", &minigpu::torch_backend::device_count);
    m.def("get_device", &minigpu::torch_backend::get_device);
    m.def("set_device", &minigpu::torch_backend::set_device);
    m.def(
        "connect",
        &minigpu::torch_backend::connect,
        pybind11::arg("port"),
        pybind11::arg("baud") = 115200,
        pybind11::arg("memory_size") = 65536);
    m.def("disconnect", &minigpu::torch_backend::disconnect);
}
