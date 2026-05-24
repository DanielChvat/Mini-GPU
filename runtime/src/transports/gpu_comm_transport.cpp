#include "transports/gpu_comm_transport.hpp"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

namespace minigpu::transports {

namespace {

constexpr int kRetries = 3;
constexpr std::uint32_t kDefaultAddressUnitBytes = 1u;

std::uint32_t status_pc(std::uint32_t status_word) noexcept {
    return (status_word >> 20) & 0xfffu;
}

std::uint32_t status_retired(std::uint32_t status_word) noexcept {
    return (status_word >> 8) & 0xfffu;
}

std::string device_fault_message(std::uint32_t status_word, const char *fault) {
    std::ostringstream out;
    out << "Mini-GPU device fault: " << fault
        << " status=0x" << std::hex << std::setw(8) << std::setfill('0') << status_word
        << " pc=0x" << std::setw(3) << status_pc(status_word)
        << " retired=" << std::dec << status_retired(status_word);
    return out.str();
}

/* Return true when an address range fits in the current 16-bit COM address field. */
bool fits_u16(DeviceAddress addr, std::size_t size = 0) {
    if (addr > std::numeric_limits<std::uint16_t>::max()) {
        return false;
    }
    return size <= static_cast<std::size_t>(
        std::numeric_limits<std::uint16_t>::max() - addr) + 1u;
}

/* Return true when a byte range can be expressed as protocol word addresses. */
bool aligned_range(DeviceAddress addr, std::size_t size, std::uint32_t unit_bytes) {
    return unit_bytes != 0 && (addr % unit_bytes) == 0 && (size % unit_bytes) == 0;
}

/* Return the largest payload size that preserves 32-bit memory-word alignment. */
std::uint16_t max_aligned_payload(std::uint32_t unit_bytes) {
    (void)unit_bytes;
    if (COM_WORD_BYTES == 0 || COM_WORD_BYTES > COM_MAX_PAYLOAD) {
        return 0;
    }
    return static_cast<std::uint16_t>(COM_MAX_ALIGNED_PAYLOAD);
}

/* Round a transfer length up to the RTL memory word size. */
std::uint16_t padded_payload_len(std::uint16_t len) {
    return static_cast<std::uint16_t>(
        (static_cast<std::uint32_t>(len) + COM_WORD_BYTES - 1u) &
        ~(COM_WORD_BYTES - 1u));
}

/* Convert a byte address into a protocol address. */
Status protocol_addr(DeviceAddress byte_addr, std::uint32_t unit_bytes, std::uint16_t *addr) {
    if (!addr || unit_bytes == 0 || (byte_addr % unit_bytes) != 0) {
        return Status::BadArgument;
    }

    const DeviceAddress protocol_address = byte_addr / unit_bytes;
    if (!fits_u16(protocol_address)) {
        return Status::OutOfRange;
    }

    *addr = static_cast<std::uint16_t>(protocol_address);
    return Status::Ok;
}

/* Store a 32-bit value in the byte order reconstructed by the RTL RX word packer. */
void put_u32_rtl_rx(std::uint8_t *dst, std::uint32_t value) {
    dst[0] = static_cast<std::uint8_t>(value & 0xffu);
    dst[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    dst[2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
    dst[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

/* Load a 32-bit value from big-endian packet payload order. */
std::uint32_t get_u32_be(const std::uint8_t *src) {
    return (static_cast<std::uint32_t>(src[0]) << 24) |
           (static_cast<std::uint32_t>(src[1]) << 16) |
           (static_cast<std::uint32_t>(src[2]) << 8) |
           static_cast<std::uint32_t>(src[3]);
}

/* Load a 32-bit value from little-endian packet payload order. */
std::uint32_t get_u32_le(const std::uint8_t *src) {
    return static_cast<std::uint32_t>(src[0]) |
           (static_cast<std::uint32_t>(src[1]) << 8) |
           (static_cast<std::uint32_t>(src[2]) << 16) |
           (static_cast<std::uint32_t>(src[3]) << 24);
}

/* Convert host-order 32-bit words into RTL RX packet payload words. */
void host_words_to_packet(std::uint8_t *dst, const std::uint8_t *src, std::size_t size) {
    for (std::size_t offset = 0; offset < size; offset += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, src + offset, sizeof(word));
        put_u32_rtl_rx(dst + offset, word);
    }
}

/* Convert big-endian packet payload words into host-order 32-bit words. */
void packet_words_to_host(std::uint8_t *dst, const std::uint8_t *src, std::size_t size) {
    for (std::size_t offset = 0; offset < size; offset += 4) {
        const std::uint32_t word = get_u32_be(src + offset);
        std::memcpy(dst + offset, &word, sizeof(word));
    }
}

/* Read and validate an ACK packet for a command address. */
Status expect_ack(com_dev_t *dev, std::uint16_t addr) {
    std::uint8_t ack_buf[COM_OVERHEAD];
    com_packet_t ack_packet;
    com_err_t err = com_recv_raw(dev, ack_buf, COM_OVERHEAD);
    if (err != COM_OK) {
        return status_from_com(err);
    }

    err = com_parse_packet(ack_buf, COM_OVERHEAD, &ack_packet);
    if (err != COM_OK) {
        return status_from_com(err);
    }

    if (ack_packet.cmd != COM_CMD_ACK || ack_packet.addr != addr || ack_packet.len != 0) {
        return Status::Transport;
    }

    return Status::Ok;
}

/* Send an ACK packet for a response address. */
Status send_ack(com_dev_t *dev, std::uint16_t addr) {
    std::uint8_t packet[COM_OVERHEAD];
    int packet_len = com_build_packet(packet, COM_CMD_ACK, addr, nullptr, 0);
    if (packet_len < 0) {
        return status_from_com(static_cast<com_err_t>(packet_len));
    }
    return status_from_com(com_send_raw(dev, packet, static_cast<std::size_t>(packet_len)));
}

/* Send one command packet. */
Status send_packet(
    com_dev_t *dev,
    com_cmd_t cmd,
    std::uint16_t addr,
    const std::uint8_t *payload,
    std::uint16_t payload_len) {
    std::uint8_t packet[COM_MAX_PACKET];
    int packet_len = com_build_packet(packet, cmd, addr, payload, payload_len);
    if (packet_len < 0) {
        return status_from_com(static_cast<com_err_t>(packet_len));
    }

    Status last_status = Status::Transport;
    for (int attempt = 0; attempt < kRetries; ++attempt) {
        last_status = status_from_com(
            com_send_raw(dev, packet, static_cast<std::size_t>(packet_len)));
        if (last_status != Status::Ok) {
            continue;
        }

        last_status = expect_ack(dev, addr);
        if (last_status == Status::Ok) {
            return Status::Ok;
        }
    }

    return last_status;
}

/* Write an arbitrary byte span with a packet command that accepts payload chunks. */
Status write_chunked_command(
    com_dev_t *dev,
    com_cmd_t cmd,
    DeviceAddress dst_addr,
    const void *src,
    std::size_t size,
    std::uint32_t unit_bytes = kDefaultAddressUnitBytes) {
    if (!dev || (!src && size)) {
        return Status::BadArgument;
    }
    if (unit_bytes != kDefaultAddressUnitBytes) {
        return Status::BadArgument;
    }
    if (!aligned_range(dst_addr, size, unit_bytes)) {
        return Status::BadArgument;
    }
    std::uint16_t start_addr = 0;
    Status addr_status = protocol_addr(dst_addr, unit_bytes, &start_addr);
    if (addr_status != Status::Ok) {
        return addr_status;
    }

    const auto *bytes = static_cast<const std::uint8_t *>(src);
    std::uint8_t payload[COM_MAX_PAYLOAD];
    std::size_t offset = 0;
    while (offset < size) {
        const std::size_t remaining = size - offset;
        const std::uint16_t max_payload = max_aligned_payload(unit_bytes);
        if (max_payload == 0) {
            return Status::BadArgument;
        }
        const auto chunk_len = static_cast<std::uint16_t>(
            remaining > max_payload ? max_payload : remaining);
        std::uint16_t chunk_addr = 0;
        Status chunk_addr_status =
            protocol_addr(dst_addr + static_cast<DeviceAddress>(offset), unit_bytes, &chunk_addr);
        if (chunk_addr_status != Status::Ok) {
            return chunk_addr_status;
        }

        const std::uint16_t packet_payload_len = padded_payload_len(chunk_len);
        std::memset(payload, 0, packet_payload_len);
        std::memcpy(payload, bytes + offset, chunk_len);

        Status status = send_packet(dev, cmd, chunk_addr, payload, packet_payload_len);
        if (status != Status::Ok) {
            return status;
        }

        offset += chunk_len;
    }

    return Status::Ok;
}

Status read_packet(
    com_dev_t *dev,
    com_cmd_t cmd,
    std::uint16_t addr,
    const std::uint8_t *payload,
    std::uint16_t payload_len,
    com_packet_t *response);

/* Read an arbitrary byte span with a command that returns payload chunks. */
Status read_chunked_command(
    com_dev_t *dev,
    com_cmd_t cmd,
    DeviceAddress src_addr,
    void *dst,
    std::size_t size,
    std::uint32_t unit_bytes = kDefaultAddressUnitBytes) {
    if (!dev || (!dst && size)) {
        return Status::BadArgument;
    }
    if (unit_bytes != kDefaultAddressUnitBytes) {
        return Status::BadArgument;
    }
    if (!aligned_range(src_addr, size, unit_bytes)) {
        return Status::BadArgument;
    }
    std::uint16_t start_addr = 0;
    Status addr_status = protocol_addr(src_addr, unit_bytes, &start_addr);
    if (addr_status != Status::Ok) {
        return addr_status;
    }

    auto *bytes = static_cast<std::uint8_t *>(dst);
    std::size_t offset = 0;
    while (offset < size) {
        const std::size_t remaining = size - offset;
        const std::uint16_t max_payload = max_aligned_payload(unit_bytes);
        if (max_payload == 0) {
            return Status::BadArgument;
        }
        const auto chunk_len = static_cast<std::uint16_t>(
            remaining > max_payload ? max_payload : remaining);
        std::uint16_t chunk_addr = 0;
        Status chunk_addr_status =
            protocol_addr(src_addr + static_cast<DeviceAddress>(offset), unit_bytes, &chunk_addr);
        if (chunk_addr_status != Status::Ok) {
            return chunk_addr_status;
        }

        com_packet_t response;
        const std::uint16_t packet_payload_len = padded_payload_len(chunk_len);
        std::uint8_t payload[COM_MAX_ALIGNED_PAYLOAD] = {};
        Status status = read_packet(
            dev, cmd, chunk_addr, payload, packet_payload_len, &response);
        if (status != Status::Ok) {
            return status;
        }
        if (response.cmd != cmd ||
            response.addr != chunk_addr ||
            response.len != packet_payload_len) {
            return Status::Transport;
        }

        std::memcpy(bytes + offset, response.payload, chunk_len);
        offset += chunk_len;
    }

    return Status::Ok;
}

/* Send a request packet and read one response packet. */
Status read_packet(
    com_dev_t *dev,
    com_cmd_t cmd,
    std::uint16_t addr,
    const std::uint8_t *payload,
    std::uint16_t payload_len,
    com_packet_t *response) {
    std::uint8_t tx_packet[COM_MAX_PACKET];
    std::uint8_t rx_packet[COM_MAX_PACKET];

    int packet_len = payload
        ? com_build_packet(tx_packet, cmd, addr, payload, payload_len)
        : com_build_request_packet(tx_packet, cmd, addr, payload_len);
    if (packet_len < 0) {
        return status_from_com(static_cast<com_err_t>(packet_len));
    }

    Status last_status = Status::Transport;
    for (int attempt = 0; attempt < kRetries; ++attempt) {
        last_status = status_from_com(
            com_send_raw(dev, tx_packet, static_cast<std::size_t>(packet_len)));
        if (last_status != Status::Ok) {
            continue;
        }

        com_err_t err = com_recv_raw(dev, rx_packet, COM_HEADER_SIZE);
        if (err != COM_OK) {
            last_status = status_from_com(err);
            continue;
        }
        if (rx_packet[0] != COM_SOF) {
            last_status = Status::Transport;
            continue;
        }

        const std::uint16_t response_len =
            (static_cast<std::uint16_t>(rx_packet[4]) << 8) |
            static_cast<std::uint16_t>(rx_packet[5]);
        if (response_len > COM_MAX_PAYLOAD) {
            last_status = Status::Transport;
            continue;
        }

        err = com_recv_raw(
            dev, rx_packet + COM_HEADER_SIZE,
            static_cast<std::size_t>(response_len) + 1u);
        if (err != COM_OK) {
            last_status = status_from_com(err);
            continue;
        }

        err = com_parse_packet(
            rx_packet, COM_OVERHEAD + static_cast<std::size_t>(response_len),
            response);
        if (err != COM_OK) {
            last_status = status_from_com(err);
            continue;
        }

        return Status::Ok;
    }

    return last_status;
}

} // namespace

/* Convert gpu_comm status codes into runtime status codes. */
Status status_from_com(com_err_t err) noexcept {
    switch (err) {
        case COM_OK:
            return Status::Ok;
        case COM_ERR_BAD_ARG:
        case COM_ERR_PAYLOAD_SIZE:
            return Status::BadArgument;
        case COM_ERR_TIMEOUT:
            return Status::Timeout;
        case COM_ERR_SHORT:
        case COM_ERR_CRC:
        case COM_ERR_NO_SOF:
        case COM_ERR_IO:
        case COM_ERR_OPEN:
        case COM_ERR_FPGA:
        default:
            return Status::Transport;
    }
}

/* Write bytes into data memory using the gpu_comm data-write command. */
Status gpu_comm_write_data(
    com_dev_t *dev, DeviceAddress dst_addr, const void *src, std::size_t size) {
    if (!fits_u16(dst_addr, size)) {
        return Status::OutOfRange;
    }
    return status_from_com(com_write_data(
        dev,
        static_cast<std::uint16_t>(dst_addr),
        static_cast<const std::uint8_t *>(src),
        size));
}

/* Read bytes from data memory using the gpu_comm data-read command. */
Status gpu_comm_read_data(
    com_dev_t *dev, DeviceAddress src_addr, void *dst, std::size_t size) {
    return read_chunked_command(dev, COM_CMD_READ_DATA, src_addr, dst, size);
}

/* Write encoded instruction bytes using the gpu_comm program-write command. */
Status gpu_comm_write_program(
    com_dev_t *dev, DeviceAddress dst_addr, const void *src, std::size_t size) {
    if ((size % COM_WORD_BYTES) != 0u) {
        return Status::BadArgument;
    }

    std::vector<std::uint8_t> packet_words(size);
    host_words_to_packet(
        packet_words.data(), static_cast<const std::uint8_t *>(src), size);
    return write_chunked_command(
        dev, COM_CMD_WRITE_PROGRAM, dst_addr, packet_words.data(), packet_words.size());
}

/* Validate loaded kernel via its ID. */
Status gpu_validate_kernel(
    com_dev_t *dev, std::uint8_t kernel_id) {
    if (!dev) {
        return Status::BadArgument;
    }

    return send_packet(dev, COM_CMD_VALIDATE, kernel_id, nullptr, 0);
}

/* Write constant bytes using the configured constant-memory transport behavior. */
Status gpu_comm_write_constants(
    com_dev_t *dev,
    const GpuCommTransportConfig &config,
    DeviceAddress dst_addr,
    const void *src,
    std::size_t size) {
    if ((size % COM_WORD_BYTES) != 0u) {
        return Status::BadArgument;
    }

    if (config.constants_use_data_memory) {
        return write_chunked_command(
            dev, COM_CMD_WRITE_DATA, dst_addr, src, size, config.address_unit_bytes);
    }

    if (config.address_unit_bytes != kDefaultAddressUnitBytes) {
        return Status::BadArgument;
    }

    std::vector<std::uint8_t> packet_words(size);
    host_words_to_packet(
        packet_words.data(), static_cast<const std::uint8_t *>(src), size);
    return write_chunked_command(
        dev, COM_CMD_WRITE_CONSTANTS, dst_addr, packet_words.data(), packet_words.size());
}

/* Send a kernel launch command with launch dimensions in the payload. */
Status gpu_comm_launch(
    com_dev_t *dev,
    DeviceAddress base_pc,
    std::uint32_t grid_dim,
    std::uint32_t block_dim,
    std::uint32_t active_mask) {
    if (!dev) {
        return Status::BadArgument;
    }
    if (!fits_u16(base_pc)) {
        return Status::OutOfRange;
    }

    std::uint8_t payload[12];
    put_u32_rtl_rx(payload, grid_dim);
    put_u32_rtl_rx(payload + 4, block_dim);
    put_u32_rtl_rx(payload + 8, active_mask);
    return send_packet(
        dev, COM_CMD_LAUNCH, static_cast<std::uint16_t>(base_pc),
        payload, sizeof(payload));
}

/* Read the current device status word. */
Status gpu_comm_read_status(com_dev_t *dev, std::uint32_t *status) {
    if (!dev || !status) {
        return Status::BadArgument;
    }

    std::uint8_t payload[4] = {};
    com_packet_t response;
    Status read_status = read_packet(dev, COM_CMD_READ_STATUS, 0, payload, sizeof(payload), &response);
    if (read_status != Status::Ok) {
        return read_status;
    }
    if (response.cmd != COM_CMD_READ_STATUS || response.len < 4) {
        return Status::Transport;
    }

    *status = get_u32_le(response.payload);
    return Status::Ok;
}

/* Poll the device status word until the done mask is observed. */
Status gpu_comm_wait(
    com_dev_t *dev, const GpuCommTransportConfig &config, std::uint32_t timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(timeout_ms);

    while (true) {
        std::uint32_t status_word = 0;
        Status status = gpu_comm_read_status(dev, &status_word);
        if (status != Status::Ok) {
            return status;
        }
        if ((status_word & config.status_error_mask) != 0u) {
            if ((status_word & config.status_divide_by_zero_bit) != 0u) {
                throw Error(
                    Status::DivideByZero,
                    device_fault_message(status_word, "divide by zero"));
            }
            if ((status_word & config.status_unsupported_bit) != 0u) {
                throw Error(
                    Status::Unsupported,
                    device_fault_message(status_word, "unsupported instruction"));
            }
            if ((status_word & config.status_error_bit) != 0u) {
                throw Error(
                    Status::Transport,
                    device_fault_message(status_word, "core error"));
            }
            throw Error(
                Status::Transport,
                device_fault_message(status_word, "unknown status error"));
        }
        if ((status_word & config.status_done_mask) != 0u) {
            return Status::Ok;
        }
        if (timeout_ms != 0 &&
            std::chrono::steady_clock::now() - start >= timeout) {
            return Status::Timeout;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/* Create runtime transport callbacks backed by a gpu_comm device handle. */
Transport make_gpu_comm_transport(com_dev_t *dev, GpuCommTransportConfig config) {
    Transport transport;
    transport.write_data =
        [dev, config](DeviceAddress dst_addr, const void *src, std::size_t size) {
            if (config.address_unit_bytes != kDefaultAddressUnitBytes) {
                return write_chunked_command(
                    dev, COM_CMD_WRITE_DATA, dst_addr, src, size, config.address_unit_bytes);
            }
            return gpu_comm_write_data(dev, dst_addr, src, size);
    };
    transport.read_data =
        [dev, config](DeviceAddress src_addr, void *dst, std::size_t size) {
            return read_chunked_command(
                dev, COM_CMD_READ_DATA, src_addr, dst, size, config.address_unit_bytes);
    };
    transport.write_program =
        [dev, config](DeviceAddress dst_addr, const void *src, std::size_t size) {
            if (config.address_unit_bytes != kDefaultAddressUnitBytes) {
                return Status::BadArgument;
            }
            return gpu_comm_write_program(dev, dst_addr, src, size);
    };
    transport.validate_kernel =
        [dev](std::uint8_t kernel_id) {
            return gpu_validate_kernel(dev, kernel_id);
        };
    transport.write_constants =
        [dev, config](DeviceAddress dst_addr, const void *src, std::size_t size) {
            return gpu_comm_write_constants(dev, config, dst_addr, src, size);
        };
    transport.launch =
        [dev](
            DeviceAddress base_pc,
            std::uint32_t grid_dim,
            std::uint32_t block_dim,
            std::uint32_t active_mask) {
            return gpu_comm_launch(dev, base_pc, grid_dim, block_dim, active_mask);
        };
    transport.wait = [dev, config](std::uint32_t timeout_ms) {
        return gpu_comm_wait(dev, config, timeout_ms);
    };
    return transport;
}

} // namespace minigpu::transports
