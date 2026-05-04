#include "minigpu_torch.hpp"

#include "minigpu_runtime.hpp"

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

} // namespace

minigpu::Context &runtime_context() {
    if (!state.context) {
        throw std::runtime_error(
            "Mini-GPU runtime context is not connected to gpu_comm yet");
    }
    return *state.context;
}

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

} // namespace minigpu::torch_backend
