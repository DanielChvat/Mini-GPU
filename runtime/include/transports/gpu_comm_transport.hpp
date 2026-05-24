#ifndef MINIGPU_TRANSPORTS_GPU_COMM_TRANSPORT_HPP
#define MINIGPU_TRANSPORTS_GPU_COMM_TRANSPORT_HPP

#include "minigpu_runtime.hpp"

extern "C" {
#include "com.h"
}

#include <cstdint>

namespace minigpu::transports {

/* Options for the gpu_comm UART transport adapter. */
struct GpuCommTransportConfig {
    bool constants_use_data_memory = false;
    std::uint32_t status_done_mask = 0x1u;
    std::uint32_t status_error_mask = 0x1cu;
    std::uint32_t status_error_bit = 0x04u;
    std::uint32_t status_unsupported_bit = 0x08u;
    std::uint32_t status_divide_by_zero_bit = 0x10u;
    /* Number of bytes represented by one protocol address step. */
    std::uint32_t address_unit_bytes = 1u;
};

/* Convert a gpu_comm error code into a runtime status code. */
Status status_from_com(com_err_t err) noexcept;

/* Write bytes to data memory through gpu_comm. */
Status gpu_comm_write_data(
    com_dev_t *dev, DeviceAddress dst_addr, const void *src, std::size_t size);

/* Read bytes from data memory through gpu_comm. */
Status gpu_comm_read_data(
    com_dev_t *dev, DeviceAddress src_addr, void *dst, std::size_t size);

/* Write encoded program bytes through gpu_comm. */
Status gpu_comm_write_program(
    com_dev_t *dev, DeviceAddress dst_addr, const void *src, std::size_t size);

/* Write constant bytes through gpu_comm. */
Status gpu_comm_write_constants(
    com_dev_t *dev, const GpuCommTransportConfig &config,
    DeviceAddress dst_addr, const void *src, std::size_t size);

/* Validate loaded kernel via it's ID*/
Status gpu_validate_kernel(
    com_dev_t *dev, std::uint8_t kernel_id);

/* Send a launch command through gpu_comm. */
Status gpu_comm_launch(
    com_dev_t *dev,
    DeviceAddress base_pc,
    std::uint32_t grid_dim,
    std::uint32_t block_dim,
    std::uint32_t active_mask);

/* Read the device status word through gpu_comm. */
Status gpu_comm_read_status(com_dev_t *dev, std::uint32_t *status);

/* Poll device status until the done bit is observed or timeout expires. */
Status gpu_comm_wait(
    com_dev_t *dev, const GpuCommTransportConfig &config, std::uint32_t timeout_ms);

/* Build a minigpu::Transport backed by a gpu_comm device handle. */
Transport make_gpu_comm_transport(
    com_dev_t *dev, GpuCommTransportConfig config = GpuCommTransportConfig{});

} // namespace minigpu::transports

#endif
