#include "minigpu_torch.hpp"

#include "minigpu_kernels.hpp"
#include "minigpu_runtime.hpp"
#include "transports/gpu_comm_transport.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <memory>
#include <mutex>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace minigpu::torch_backend {

namespace {

/* Process-wide state shared by PyTorch operator stubs. */
struct BackendState {
    bool initialized = false;
    int current_device = 0;
    com_dev_t *dev = nullptr;
    std::unique_ptr<minigpu::Context> context;
};

std::mutex state_mutex;
BackendState state;

class MiniGpuPrivateUse1Hooks final : public at::PrivateUse1HooksInterface {
public:
    bool isBuilt() const override {
        return true;
    }

    bool isAvailable() const override {
        return state.context != nullptr;
    }

    const at::Generator &getDefaultGenerator(c10::DeviceIndex device_index) const override {
        (void)device_index;
        return at::detail::getDefaultCPUGenerator();
    }

    at::Device getDeviceFromPtr(void *data) const override {
        (void)data;
        return at::Device(c10::DeviceType::PrivateUse1, 0);
    }

    c10::Allocator *getPinnedMemoryAllocator() const override {
        return nullptr;
    }

    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        return device_index == 0 && state.context != nullptr;
    }

    void resizePrivateUse1Bytes(const c10::Storage &storage, std::size_t newsize) const override {
        (void)storage;
        if (newsize != 0) {
            throw std::runtime_error("Mini-GPU storage resize is not supported yet");
        }
    }
};

MiniGpuPrivateUse1Hooks hooks;

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

    if (!at::isPrivateUse1HooksRegistered()) {
        at::RegisterPrivateUse1HooksInterface(&hooks);
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

void connect(const std::string &port, std::uint32_t baud, std::uint32_t memory_size) {
    std::lock_guard<std::mutex> lock(state_mutex);
    if (state.context) {
        return;
    }

    com_dev_t *dev = open_com(port.c_str(), static_cast<int>(baud), 5000);
    if (!dev) {
        throw std::runtime_error("failed to open Mini-GPU serial device: " + port);
    }

    minigpu::Config config;
    config.memory_base = 0;
    config.memory_size = memory_size;
    if (const char *program_bytes = std::getenv("MINIGPU_PROGRAM_MEMORY_BYTES")) {
        if (program_bytes[0] != '\0') {
            config.program_memory_size = static_cast<std::size_t>(std::stoul(program_bytes));
        }
    }
    config.default_alignment = 4;
    config.transport = minigpu::transports::make_gpu_comm_transport(dev);

    try {
        if (!at::isPrivateUse1HooksRegistered()) {
            at::RegisterPrivateUse1HooksInterface(&hooks);
        }
        state.context = std::make_unique<minigpu::Context>(std::move(config));
        minigpu::kernels::register_builtin_kernels(*state.context);
        state.dev = dev;
        state.initialized = true;
    } catch (...) {
        close_com(dev);
        throw;
    }
}

void disconnect() {
    std::lock_guard<std::mutex> lock(state_mutex);
    state.context.reset();
    if (state.dev) {
        close_com(state.dev);
        state.dev = nullptr;
    }
}

} // namespace minigpu::torch_backend
