#include "minigpu_torch.hpp"

#include "transports/gpu_comm_transport.hpp"

namespace minigpu::torch_backend {

minigpu::Status map_com_status(int err) {
    return minigpu::transports::status_from_com(static_cast<com_err_t>(err));
}

minigpu::Transport make_gpu_comm_transport(void *dev) {
    return minigpu::transports::make_gpu_comm_transport(static_cast<com_dev_t *>(dev));
}

} // namespace minigpu::torch_backend
