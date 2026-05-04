#include "minigpu_torch.hpp"

#include "minigpu_runtime.hpp"

namespace minigpu::torch_backend {

minigpu::Status map_com_status(int err) {
    return err == 0 ? minigpu::Status::Ok : minigpu::Status::Transport;
}

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

} // namespace minigpu::torch_backend
