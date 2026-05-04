#ifndef MINIGPU_RUNTIME_HPP
#define MINIGPU_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace minigpu {

using DeviceAddress = std::uint32_t;

/* Runtime status codes used by Mini-GPU helper functions and exceptions. */
enum class Status {
    Ok = 0,
    BadArgument,
    NoMemory,
    OutOfRange,
    NotFound,
    Transport,
    Timeout,
    Unsupported,
};

/* Convert a runtime status code into a human-readable string. */
std::string_view status_message(Status status) noexcept;

/* Exception wrapper for Mini-GPU runtime failures. */
class Error : public std::runtime_error {
public:
    /* Build an exception with status_message(status) as the message. */
    explicit Error(Status status);

    /* Return the original runtime status code. */
    Status status() const noexcept;

private:
    Status status_;
};

/* Throw minigpu::Error if a runtime status code is not Status::Ok. */
void check(Status status);

/* Transport callback table used by Context for all device I/O. */
struct Transport {
    /* Write bytes into Mini-GPU data/global memory. */
    std::function<Status(DeviceAddress dst_addr, const void *src, std::size_t size)>
        write_data;

    /* Read bytes from Mini-GPU data/global memory. */
    std::function<Status(DeviceAddress src_addr, void *dst, std::size_t size)>
        read_data;

    /* Write encoded instruction words into program memory. */
    std::function<Status(DeviceAddress dst_addr, const void *src, std::size_t size)>
        write_program;

    /* Write constant data for LDC-style kernel operands. */
    std::function<Status(DeviceAddress dst_addr, const void *src, std::size_t size)>
        write_constants;

    /* Start a kernel after program/data setup is complete. */
    std::function<Status(
        DeviceAddress base_pc, std::uint32_t grid_dim, std::uint32_t block_dim,
        std::uint32_t active_mask)>
        launch;

    /* Optional callback used to wait for kernel completion. */
    std::function<Status(std::uint32_t timeout_ms)> wait;
};

/* Runtime creation parameters, including device memory range and transport. */
struct Config {
    DeviceAddress memory_base = 0;
    std::size_t memory_size = 0;
    std::size_t default_alignment = 4;
    Transport transport;
};

/* Kernel launch descriptor used by Context::launch_kernel. */
struct Kernel {
    const void *program = nullptr;
    std::size_t program_size = 0;
    DeviceAddress program_addr = 0;
    const void *constants = nullptr;
    std::size_t constants_size = 0;
    DeviceAddress constants_addr = 0;
    DeviceAddress base_pc = 0;
    std::uint32_t grid_dim = 1;
    std::uint32_t block_dim = 1;
    std::uint32_t active_mask = 0xffffffffu;
    std::uint32_t timeout_ms = 0;
};

class Context;

/* RAII owner for one device allocation inside a Context. */
class DeviceBuffer {
public:
    /* Create an empty buffer handle. */
    DeviceBuffer() = default;

    /* Take ownership of an existing device allocation. */
    DeviceBuffer(Context *context, DeviceAddress addr, std::size_t size);

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    /* Move ownership from another DeviceBuffer. */
    DeviceBuffer(DeviceBuffer &&other) noexcept;

    /* Free the current allocation, then move ownership from another buffer. */
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

    /* Free the owned device allocation, if any. */
    ~DeviceBuffer();

    /* Return the device address of this allocation. */
    DeviceAddress addr() const noexcept;

    /* Return the allocation size requested by the caller. */
    std::size_t size() const noexcept;

    /* Return true when this handle owns an allocation. */
    explicit operator bool() const noexcept;

    /* Release ownership without freeing and return the device address. */
    DeviceAddress release() noexcept;

    /* Free the owned device allocation and clear this handle. */
    void reset() noexcept;

private:
    void move_from(DeviceBuffer &other) noexcept;

    Context *context_ = nullptr;
    DeviceAddress addr_ = 0;
    std::size_t size_ = 0;
};

/* Runtime context that owns allocator state and transport callbacks. */
class Context {
public:
    /* Create a runtime context from Mini-GPU configuration. */
    explicit Context(Config config);

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    /* Move ownership of the runtime context. */
    Context(Context &&other) noexcept;

    /* Destroy the current context, then move ownership from another context. */
    Context &operator=(Context &&other) noexcept;

    /* Destroy the runtime context and allocator state. */
    ~Context();

    /* Allocate device memory using the context default alignment. */
    DeviceBuffer device_malloc(std::size_t size);

    /* Allocate device memory using an explicit byte alignment. */
    DeviceBuffer device_malloc_aligned(std::size_t size, std::size_t alignment);

    /* Free a device allocation previously returned by device_malloc*. */
    Status device_free(DeviceAddress addr) noexcept;

    /* Copy host bytes into an allocated device memory range. */
    void copy_to_device(DeviceAddress dst_addr, const void *src, std::size_t size);

    /* Copy bytes from an allocated device memory range into host memory. */
    void copy_from_device(void *dst, DeviceAddress src_addr, std::size_t size);

    /* Write encoded program bytes through the configured transport. */
    void write_program(DeviceAddress dst_addr, const void *program, std::size_t size);

    /* Write constant-memory bytes through the configured transport. */
    void write_constants(
        DeviceAddress dst_addr, const void *constants, std::size_t size);

    /* Optionally write program/constants, launch the kernel, and wait if supported. */
    void launch_kernel(const Kernel &kernel);

    /* Return the total managed device memory size in bytes. */
    std::size_t memory_size() const noexcept;

    /* Return the number of currently free managed device memory bytes. */
    std::size_t memory_free() const noexcept;

private:
    /* One contiguous allocator entry in the managed device-memory range.
     * offset is an absolute device address, size is bytes, and free marks
     * whether this block can satisfy a future allocation. */
    struct Block {
        DeviceAddress offset = 0;
        std::size_t size = 0;
        bool free = true;
    };

    DeviceAddress memory_base_ = 0;
    std::size_t memory_size_ = 0;
    std::size_t default_alignment_ = 4;
    Transport transport_;
    std::vector<Block> blocks_;

    /* Return true when the entire address range belongs to one live allocation. */
    bool range_is_allocated(DeviceAddress addr, std::size_t size) const noexcept;

    /* Merge adjacent free allocator blocks after a free operation. */
    void coalesce_free_blocks();
};

} // namespace minigpu

#endif
