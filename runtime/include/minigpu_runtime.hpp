#ifndef MINIGPU_RUNTIME_HPP
#define MINIGPU_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
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
    DivideByZero,
};

/* Convert a runtime status code into a human-readable string. */
std::string_view status_message(Status status) noexcept;

/* Exception wrapper for Mini-GPU runtime failures. */
class Error : public std::runtime_error {
public:
    /* Build an exception with status_message(status) as the message. */
    explicit Error(Status status);

    /* Build an exception with device/runtime context in the message. */
    Error(Status status, std::string message);

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
    std::size_t program_memory_size = 4096u * sizeof(std::uint32_t);
    std::size_t default_alignment = 4;
    Transport transport;
    bool enable_logging = false;
    std::string log_path;
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

/* A precompiled instruction stream that can be launched by registry name. */
struct PrecompiledKernel {
    std::string name;
    std::string program_id;
    std::string op;
    std::string dtype;
    const void *program = nullptr;
    std::size_t program_size = 0;
    std::vector<std::uint8_t> program_bytes;
    DeviceAddress program_addr = 0;
    DeviceAddress base_pc = 0;
    DeviceAddress entry_pc = 0;
    std::uint32_t default_grid_dim = 1;
    std::uint32_t default_block_dim = 4;
    std::uint32_t default_active_mask = 0xffffffffu;
    std::uint32_t default_timeout_ms = 0;
};

/* Launch dimensions used when overriding a registered kernel's defaults. */
struct LaunchConfig {
    std::uint32_t grid_dim = 1;
    std::uint32_t block_dim = 4;
    std::uint32_t active_mask = 0xffffffffu;
    std::uint32_t timeout_ms = 0;
};

/* One 32-bit value written into constant memory before a kernel launch. */
struct KernelArg {
    std::uint32_t value = 0;

    /* Build a raw 32-bit constant argument. */
    static KernelArg u32(std::uint32_t value) noexcept;

    /* Build a raw FP32 constant argument. */
    static KernelArg f32(float value) noexcept;

    /* Build a word-address pointer argument from a byte-addressed allocation. */
    static KernelArg device_ptr(DeviceAddress byte_addr, std::uint32_t bytes_per_word = 4);
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

    /* Register or replace a precompiled kernel by name. */
    void register_kernel(PrecompiledKernel kernel);

    /* Load a precompiled kernel file and register it by name. */
    void register_kernel_from_file(
        std::string name,
        std::string_view path,
        DeviceAddress program_addr = 0,
        DeviceAddress base_pc = 0,
        LaunchConfig defaults = LaunchConfig{});

    /* Return a registered kernel by name, or nullptr if it is not registered. */
    const PrecompiledKernel *find_kernel(std::string_view name) const noexcept;

    /* Launch a registered kernel with a constant-memory argument list. */
    void launch_kernel(
        std::string_view name,
        const std::vector<KernelArg> &args,
        const LaunchConfig *launch_config = nullptr);

    /* Return the total managed device memory size in bytes. */
    std::size_t memory_size() const noexcept;

    /* Return the number of currently free managed device memory bytes. */
    std::size_t memory_free() const noexcept;

private:
    static constexpr std::size_t kCompilerSpillBytes = 8192;

    /* One contiguous allocator entry in the managed device-memory range.
     * offset is an absolute device address, size is bytes, and free marks
     * whether this block can satisfy a future allocation. */
    struct Block {
        DeviceAddress offset = 0;
        std::size_t size = 0;
        bool free = true;
    };

    /* One cached program artifact inside instruction memory. */
    struct ProgramCacheEntry {
        std::string name;
        DeviceAddress addr = 0;
        std::size_t size = 0;
        std::uint64_t last_used = 0;
    };

    DeviceAddress memory_base_ = 0;
    std::size_t memory_size_ = 0;
    std::size_t program_memory_size_ = 0;
    std::uint64_t program_cache_clock_ = 0;
    DeviceAddress compiler_spill_base_ = 0;
    std::size_t compiler_spill_size_ = 0;
    std::size_t default_alignment_ = 4;
    Transport transport_;
    std::vector<Block> blocks_;
    std::vector<Block> program_blocks_;
    std::vector<ProgramCacheEntry> program_cache_;
    std::vector<PrecompiledKernel> kernels_;
    bool logging_enabled_ = false;
    std::string log_path_;

    /* Return true when the entire address range belongs to one live allocation. */
    bool range_is_allocated(DeviceAddress addr, std::size_t size) const noexcept;

    /* Append one runtime event to the configured log file when logging is enabled. */
    void log_event(std::string_view event) const noexcept;

    /* Merge adjacent free allocator blocks after a free operation. */
    void coalesce_free_blocks();

    /* Merge adjacent free program-memory blocks after eviction. */
    void coalesce_program_blocks();

    /* Ensure a registered program artifact is resident in instruction memory. */
    DeviceAddress ensure_program_loaded(const PrecompiledKernel &kernel);

    /* Allocate instruction memory, evicting least-recently-used artifacts as needed. */
    DeviceAddress allocate_program_slot(std::string_view name, std::size_t size);

    /* Evict one least-recently-used cached program artifact. */
    bool evict_lru_program();
};

} // namespace minigpu

#endif
