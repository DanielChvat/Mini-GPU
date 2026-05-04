#include "minigpu_torch.hpp"

#include "minigpu_runtime.hpp"

#include <torch/extension.h>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace minigpu::torch_backend {

namespace {

/* Process-wide state shared by PyTorch operator stubs. */
struct BackendState {
    bool initialized = false;
    int current_device = 0;
    std::unique_ptr<minigpu::Context> context;
};

std::mutex state_mutex;
BackendState state;

/* Return the active runtime context. */
minigpu::Context &runtime_context() {
    if (!state.context) {
        throw std::runtime_error(
            "Mini-GPU runtime context is not connected to gpu_comm yet");
    }
    return *state.context;
}

/* Convert communication errors into runtime status values. */
minigpu::Status map_com_status(int err) {
    return err == 0 ? minigpu::Status::Ok : minigpu::Status::Transport;
}

/* Build a runtime transport for a communication device. */
minigpu::Transport make_gpu_comm_transport(void *dev) {
    (void)dev;

    minigpu::Transport transport;
    transport.write_data = [](minigpu::DeviceAddress, const void *, std::size_t) {
        return minigpu::Status::Unsupported;
    };
    transport.read_data = [](minigpu::DeviceAddress, void *, std::size_t) {
        return minigpu::Status::Unsupported;
    };
    transport.write_program = [](minigpu::DeviceAddress, const void *, std::size_t) {
        return minigpu::Status::Unsupported;
    };
    transport.write_constants = [](minigpu::DeviceAddress, const void *, std::size_t) {
        return minigpu::Status::Unsupported;
    };
    transport.launch = [](
        minigpu::DeviceAddress, std::uint32_t, std::uint32_t, std::uint32_t) {
        return minigpu::Status::Unsupported;
    };
    transport.wait = [](std::uint32_t) {
        return minigpu::Status::Unsupported;
    };
    return transport;
}

/* Throw a consistent message for operators that are intentionally skeletons. */
[[noreturn]] void unimplemented_op(const char *name) {
    throw std::runtime_error(
        std::string("Mini-GPU PyTorch op is a stub: ") + name);
}

} // namespace

void init() {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (state.initialized) {
        return;
    }

    state.initialized = true;
}

bool is_built() {
    return true;
}

bool is_available() {
    std::lock_guard<std::mutex> lock(state_mutex);
    return state.context != nullptr;
}

int device_count() {
    return is_available() ? 1 : 0;
}

int get_device() {
    std::lock_guard<std::mutex> lock(state_mutex);
    return state.current_device;
}

void set_device(int index) {
    if (index != 0) {
        throw std::runtime_error("Mini-GPU currently only supports device index 0");
    }

    std::lock_guard<std::mutex> lock(state_mutex);
    state.current_device = index;
}

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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(minigpu::torch_backend::empty));
    m.impl("copy_", TORCH_FN(minigpu::torch_backend::copy_));
    m.impl("add.Tensor", TORCH_FN(minigpu::torch_backend::add_tensor));
    m.impl("mul.Tensor", TORCH_FN(minigpu::torch_backend::mul_tensor));
    m.impl("mm", TORCH_FN(minigpu::torch_backend::mm));
    m.impl("relu", TORCH_FN(minigpu::torch_backend::relu));
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "Mini-GPU PyTorch PrivateUse1 backend stubs";
    m.def("init", &minigpu::torch_backend::init);
    m.def("is_built", &minigpu::torch_backend::is_built);
    m.def("is_available", &minigpu::torch_backend::is_available);
    m.def("device_count", &minigpu::torch_backend::device_count);
    m.def("get_device", &minigpu::torch_backend::get_device);
    m.def("set_device", &minigpu::torch_backend::set_device);
}
