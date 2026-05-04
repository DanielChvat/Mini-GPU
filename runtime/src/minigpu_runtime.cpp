#include "minigpu_runtime.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace minigpu {

namespace {

/* Round a byte count up to the requested alignment for allocation sizing. */
std::size_t align_up_size(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }

    const std::size_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

/* Round a device address up to the requested alignment for allocation placement. */
DeviceAddress align_up_addr(DeviceAddress value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }

    const auto addr_alignment = static_cast<DeviceAddress>(alignment);
    const DeviceAddress rem = value % addr_alignment;
    return rem == 0 ? value : value + (addr_alignment - rem);
}

/* Check whether addr + size would overflow the 32-bit device address space. */
bool add_overflows_addr(DeviceAddress addr, std::size_t size) {
    return size > static_cast<std::size_t>(
        std::numeric_limits<DeviceAddress>::max() - addr);
}

} // namespace

/* Convert internal Status values into short user/debug strings. */
std::string_view status_message(Status status) noexcept {
    switch (status) {
        case Status::Ok:
            return "ok";
        case Status::BadArgument:
            return "bad argument";
        case Status::NoMemory:
            return "out of device or host memory";
        case Status::OutOfRange:
            return "address range is outside Mini-GPU memory";
        case Status::NotFound:
            return "allocation was not found";
        case Status::Transport:
            return "transport operation failed";
        case Status::Timeout:
            return "operation timed out";
        case Status::Unsupported:
            return "operation is not supported by this transport";
    }

    return "unknown Mini-GPU runtime error";
}

/* Store the original status code while exposing a normal std::runtime_error. */
Error::Error(Status status)
    : std::runtime_error(std::string(status_message(status))),
      status_(status) {}

/* Return the status code that caused this runtime exception. */
Status Error::status() const noexcept {
    return status_;
}

/* Helper for callback-heavy code: turn a Status return into an exception. */
void check(Status status) {
    if (status != Status::Ok) {
        throw Error(status);
    }
}

/* Attach an existing allocation to an RAII DeviceBuffer owner. */
DeviceBuffer::DeviceBuffer(Context *context, DeviceAddress addr, std::size_t size)
    : context_(context), addr_(addr), size_(size) {}

/* Move an allocation handle without freeing the underlying device memory. */
DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept {
    move_from(other);
}

/* Free any current allocation, then take ownership from another handle. */
DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        reset();
        move_from(other);
    }
    return *this;
}

/* Automatically free the owned allocation when the handle leaves scope. */
DeviceBuffer::~DeviceBuffer() {
    reset();
}

/* Return the absolute Mini-GPU address of the allocation. */
DeviceAddress DeviceBuffer::addr() const noexcept {
    return addr_;
}

/* Return the byte size originally requested for the allocation. */
std::size_t DeviceBuffer::size() const noexcept {
    return size_;
}

/* Report whether this handle currently owns a device allocation. */
DeviceBuffer::operator bool() const noexcept {
    return context_ != nullptr;
}

/* Detach this handle from the allocation so the caller manages it manually. */
DeviceAddress DeviceBuffer::release() noexcept {
    const DeviceAddress addr = addr_;
    context_ = nullptr;
    addr_ = 0;
    size_ = 0;
    return addr;
}

/* Free the current allocation through the owning Context and clear this handle. */
void DeviceBuffer::reset() noexcept {
    if (context_) {
        (void)context_->device_free(addr_);
    }
    context_ = nullptr;
    addr_ = 0;
    size_ = 0;
}

/* Internal move helper used by the move constructor and move assignment. */
void DeviceBuffer::move_from(DeviceBuffer &other) noexcept {
    context_ = other.context_;
    addr_ = other.addr_;
    size_ = other.size_;
    other.context_ = nullptr;
    other.addr_ = 0;
    other.size_ = 0;
}

/* Build a runtime context around one managed device-memory range and transport. */
Context::Context(Config config)
    : memory_base_(config.memory_base),
      memory_size_(config.memory_size),
      default_alignment_(config.default_alignment ? config.default_alignment : 4),
      transport_(std::move(config.transport)) {
    if (memory_size_ == 0) {
        throw Error(Status::BadArgument);
    }

    if (add_overflows_addr(memory_base_, memory_size_)) {
        throw Error(Status::OutOfRange);
    }

    blocks_.push_back(Block{memory_base_, memory_size_, true});
}

/* Move allocator state and transport callbacks into a new Context object. */
Context::Context(Context &&other) noexcept
    : memory_base_(std::exchange(other.memory_base_, 0)),
      memory_size_(std::exchange(other.memory_size_, 0)),
      default_alignment_(std::exchange(other.default_alignment_, 4)),
      transport_(std::move(other.transport_)),
      blocks_(std::move(other.blocks_)) {}

/* Replace this Context with another Context's allocator and transport state. */
Context &Context::operator=(Context &&other) noexcept {
    if (this != &other) {
        memory_base_ = std::exchange(other.memory_base_, 0);
        memory_size_ = std::exchange(other.memory_size_, 0);
        default_alignment_ = std::exchange(other.default_alignment_, 4);
        transport_ = std::move(other.transport_);
        blocks_ = std::move(other.blocks_);
    }
    return *this;
}

/* Context owns only standard containers/callbacks, so default destruction is enough. */
Context::~Context() = default;

/* Allocate memory using the default alignment from Config. */
DeviceBuffer Context::device_malloc(std::size_t size) {
    return device_malloc_aligned(size, default_alignment_);
}

/* Allocate an aligned range from the simple first-fit free-list allocator. */
DeviceBuffer Context::device_malloc_aligned(
    std::size_t size, std::size_t alignment) {
    if (size == 0 || alignment == 0) {
        throw Error(Status::BadArgument);
    }

    size = align_up_size(size, alignment);

    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        Block &block = blocks_[index];
        if (!block.free) {
            continue;
        }

        const DeviceAddress aligned = align_up_addr(block.offset, alignment);
        const std::size_t padding = static_cast<std::size_t>(aligned - block.offset);
        if (padding > block.size || size > block.size - padding) {
            continue;
        }

        if (padding) {
            /* Preserve bytes before the aligned address as a smaller free block. */
            Block prefix{block.offset, padding, true};
            block.offset = aligned;
            block.size -= padding;
            blocks_.insert(blocks_.begin() + static_cast<std::ptrdiff_t>(index), prefix);
            ++index;
        }

        DeviceAddress alloc_addr = blocks_[index].offset;
        if (blocks_[index].size > size) {
            /* Preserve bytes after the allocation as a free tail block. */
            Block tail{
                blocks_[index].offset + static_cast<DeviceAddress>(size),
                blocks_[index].size - size,
                true,
            };
            blocks_[index].size = size;
            blocks_[index].free = false;
            blocks_.insert(blocks_.begin() + static_cast<std::ptrdiff_t>(index + 1), tail);
        } else {
            blocks_[index].free = false;
        }

        return DeviceBuffer(this, alloc_addr, size);
    }

    throw Error(Status::NoMemory);
}

/* Mark a previously allocated block free and merge neighboring free blocks. */
Status Context::device_free(DeviceAddress addr) noexcept {
    for (auto &block : blocks_) {
        if (!block.free && block.offset == addr) {
            block.free = true;
            coalesce_free_blocks();
            return Status::Ok;
        }
    }

    return Status::NotFound;
}

/* Validate an allocation range, then write host bytes through transport.write_data. */
void Context::copy_to_device(
    DeviceAddress dst_addr, const void *src, std::size_t size) {
    if (!src && size) {
        throw Error(Status::BadArgument);
    }
    if (!range_is_allocated(dst_addr, size)) {
        throw Error(Status::OutOfRange);
    }
    if (size == 0) {
        return;
    }
    if (!transport_.write_data) {
        throw Error(Status::Unsupported);
    }

    check(transport_.write_data(dst_addr, src, size));
}

/* Validate an allocation range, then read device bytes through transport.read_data. */
void Context::copy_from_device(
    void *dst, DeviceAddress src_addr, std::size_t size) {
    if (!dst && size) {
        throw Error(Status::BadArgument);
    }
    if (!range_is_allocated(src_addr, size)) {
        throw Error(Status::OutOfRange);
    }
    if (size == 0) {
        return;
    }
    if (!transport_.read_data) {
        throw Error(Status::Unsupported);
    }

    check(transport_.read_data(src_addr, dst, size));
}

/* Upload encoded instruction bytes through transport.write_program. */
void Context::write_program(
    DeviceAddress dst_addr, const void *program, std::size_t size) {
    if (!program && size) {
        throw Error(Status::BadArgument);
    }
    if (size == 0) {
        return;
    }
    if (!transport_.write_program) {
        throw Error(Status::Unsupported);
    }

    check(transport_.write_program(dst_addr, program, size));
}

/* Upload constants through transport.write_constants. */
void Context::write_constants(
    DeviceAddress dst_addr, const void *constants, std::size_t size) {
    if (!constants && size) {
        throw Error(Status::BadArgument);
    }
    if (size == 0) {
        return;
    }
    if (!transport_.write_constants) {
        throw Error(Status::Unsupported);
    }

    check(transport_.write_constants(dst_addr, constants, size));
}

/* Upload optional program/constants, launch the kernel, and wait if supported. */
void Context::launch_kernel(const Kernel &kernel) {
    if (kernel.program && kernel.program_size) {
        write_program(kernel.program_addr, kernel.program, kernel.program_size);
    }

    if (kernel.constants && kernel.constants_size) {
        write_constants(kernel.constants_addr, kernel.constants, kernel.constants_size);
    }

    if (!transport_.launch) {
        throw Error(Status::Unsupported);
    }

    check(transport_.launch(
        kernel.base_pc, kernel.grid_dim, kernel.block_dim, kernel.active_mask));

    if (transport_.wait) {
        check(transport_.wait(kernel.timeout_ms));
    }
}

/* Return the configured size of the managed device-memory window. */
std::size_t Context::memory_size() const noexcept {
    return memory_size_;
}

/* Sum all free-list entries currently available for future allocations. */
std::size_t Context::memory_free() const noexcept {
    std::size_t total = 0;
    for (const auto &block : blocks_) {
        if (block.free) {
            total += block.size;
        }
    }
    return total;
}

/* Ensure a transfer stays inside one live allocation before touching hardware. */
bool Context::range_is_allocated(DeviceAddress addr, std::size_t size) const noexcept {
    if (size == 0) {
        return true;
    }

    for (const auto &block : blocks_) {
        if (!block.free &&
            addr >= block.offset &&
            !add_overflows_addr(addr, size) &&
            addr + size <= block.offset + block.size) {
            return true;
        }
    }

    return false;
}

/* Collapse adjacent free blocks to reduce fragmentation after device_free. */
void Context::coalesce_free_blocks() {
    for (std::size_t i = 0; i + 1 < blocks_.size();) {
        Block &block = blocks_[i];
        const Block &next = blocks_[i + 1];
        if (block.free && next.free && block.offset + block.size == next.offset) {
            block.size += next.size;
            blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(i + 1));
        } else {
            ++i;
        }
    }
}

} // namespace minigpu
