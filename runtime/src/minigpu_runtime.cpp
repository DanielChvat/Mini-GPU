#include "minigpu_runtime.hpp"
#include "generated/kernel_id_map.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
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

/* Return the runtime log target requested by the environment. */
std::string env_log_path() {
    const char *value = std::getenv("MINIGPU_RUNTIME_LOG");
    if (!value || value[0] == '\0' || std::string_view(value) == "0") {
        return {};
    }
    if (std::string_view(value) == "1" || std::string_view(value) == "true") {
        return "/tmp/minigpu_runtime.log";
    }
    return value;
}

/* Load a whole file as bytes. */
std::vector<std::uint8_t> read_binary_file(std::string_view path) {
    std::ifstream file(std::string(path), std::ios::binary);
    if (!file) {
        throw Error(Status::NotFound);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        throw Error(Status::BadArgument);
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!file) {
        throw Error(Status::Transport);
    }
    return bytes;
}

/* Load newline-separated 32-bit hex words as host-order program words. */
std::vector<std::uint8_t> read_hex_words_file(std::string_view path) {
    std::ifstream file{std::string(path)};
    if (!file) {
        throw Error(Status::NotFound);
    }

    std::vector<std::uint8_t> bytes;
    std::string token;
    while (file >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string ignored;
            std::getline(file, ignored);
            continue;
        }

        std::uint32_t word = 0;
        std::stringstream stream(token);
        stream >> std::hex >> word;
        if (!stream || !stream.eof()) {
            throw Error(Status::BadArgument);
        }

        const auto *word_bytes = reinterpret_cast<const std::uint8_t *>(&word);
        bytes.insert(bytes.end(), word_bytes, word_bytes + sizeof(word));
    }

    if (bytes.empty()) {
        throw Error(Status::BadArgument);
    }
    return bytes;
}

/* Load either raw .bin bytes or a text .hex file of 32-bit words. */
std::vector<std::uint8_t> read_kernel_file(std::string_view path) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".hex") {
        return read_hex_words_file(path);
    }
    return read_binary_file(path);
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
            return "Mini-GPU device reported an unsupported operation";
        case Status::DivideByZero:
            return "Mini-GPU device reported divide by zero";
    }

    return "unknown Mini-GPU runtime error";
}

/* Store the original status code while exposing a normal std::runtime_error. */
Error::Error(Status status)
    : std::runtime_error(std::string(status_message(status))),
      status_(status) {}

Error::Error(Status status, std::string message)
    : std::runtime_error(std::move(message)),
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

/* Build a raw constant-memory argument. */
KernelArg KernelArg::u32(std::uint32_t value) noexcept {
    return KernelArg{value};
}

/* Build a raw FP32 constant-memory argument. */
KernelArg KernelArg::f32(float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return KernelArg{bits};
}

/* Convert a byte-addressed device pointer into the word address used by LDG/STG. */
KernelArg KernelArg::device_ptr(DeviceAddress byte_addr, std::uint32_t bytes_per_word) {
    if (bytes_per_word == 0 || (byte_addr % bytes_per_word) != 0) {
        throw Error(Status::BadArgument);
    }

    return KernelArg{byte_addr / bytes_per_word};
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
      program_memory_size_(config.program_memory_size),
      default_alignment_(config.default_alignment ? config.default_alignment : 4),
      transport_(std::move(config.transport)),
      logging_enabled_(config.enable_logging),
      log_path_(std::move(config.log_path)) {
    if (memory_size_ == 0) {
        throw Error(Status::BadArgument);
    }
    if (program_memory_size_ == 0 || (program_memory_size_ % sizeof(std::uint32_t)) != 0u) {
        throw Error(Status::BadArgument);
    }

    if (add_overflows_addr(memory_base_, memory_size_)) {
        throw Error(Status::OutOfRange);
    }

    if (log_path_.empty()) {
        log_path_ = env_log_path();
    }
    logging_enabled_ = logging_enabled_ || !log_path_.empty();
    if (logging_enabled_ && log_path_.empty()) {
        log_path_ = "/tmp/minigpu_runtime.log";
    }

    compiler_spill_size_ = align_up_size(
        std::min(kCompilerSpillBytes, std::max<std::size_t>(4, memory_size_ / 8u)),
        default_alignment_);
    if (compiler_spill_size_ >= memory_size_) {
        throw Error(Status::NoMemory);
    }

    const std::size_t managed_size = memory_size_ - compiler_spill_size_;
    compiler_spill_base_ = memory_base_ + static_cast<DeviceAddress>(managed_size);
    blocks_.push_back(Block{memory_base_, managed_size, true});
    program_blocks_.push_back(Block{0, program_memory_size_, true});
    log_event("context created");
}

/* Move allocator state and transport callbacks into a new Context object. */
Context::Context(Context &&other) noexcept
    : memory_base_(std::exchange(other.memory_base_, 0)),
      memory_size_(std::exchange(other.memory_size_, 0)),
      program_memory_size_(std::exchange(other.program_memory_size_, 0)),
      program_cache_clock_(std::exchange(other.program_cache_clock_, 0)),
      compiler_spill_base_(std::exchange(other.compiler_spill_base_, 0)),
      compiler_spill_size_(std::exchange(other.compiler_spill_size_, 0)),
      default_alignment_(std::exchange(other.default_alignment_, 4)),
      transport_(std::move(other.transport_)),
      blocks_(std::move(other.blocks_)),
      program_blocks_(std::move(other.program_blocks_)),
      program_cache_(std::move(other.program_cache_)),
      kernels_(std::move(other.kernels_)),
      logging_enabled_(std::exchange(other.logging_enabled_, false)),
      log_path_(std::move(other.log_path_)) {}

/* Replace this Context with another Context's allocator and transport state. */
Context &Context::operator=(Context &&other) noexcept {
    if (this != &other) {
        memory_base_ = std::exchange(other.memory_base_, 0);
        memory_size_ = std::exchange(other.memory_size_, 0);
        program_memory_size_ = std::exchange(other.program_memory_size_, 0);
        program_cache_clock_ = std::exchange(other.program_cache_clock_, 0);
        compiler_spill_base_ = std::exchange(other.compiler_spill_base_, 0);
        compiler_spill_size_ = std::exchange(other.compiler_spill_size_, 0);
        default_alignment_ = std::exchange(other.default_alignment_, 4);
        transport_ = std::move(other.transport_);
        blocks_ = std::move(other.blocks_);
        program_blocks_ = std::move(other.program_blocks_);
        program_cache_ = std::move(other.program_cache_);
        kernels_ = std::move(other.kernels_);
        logging_enabled_ = std::exchange(other.logging_enabled_, false);
        log_path_ = std::move(other.log_path_);
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

        std::ostringstream log;
        log << "device_malloc addr=0x" << std::hex << alloc_addr
            << " bytes=" << std::dec << size;
        log_event(log.str());
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
            std::ostringstream log;
            log << "device_free addr=0x" << std::hex << addr;
            log_event(log.str());
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

    std::ostringstream log;
    log << "copy_to_device addr=0x" << std::hex << dst_addr
        << " bytes=" << std::dec << size;
    log_event(log.str());
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

    std::ostringstream log;
    log << "copy_from_device addr=0x" << std::hex << src_addr
        << " bytes=" << std::dec << size;
    log_event(log.str());
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

    std::ostringstream log;
    log << "write_program addr=0x" << std::hex << dst_addr
        << " bytes=" << std::dec << size;
    log_event(log.str());
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

    std::ostringstream log;
    log << "write_constants addr=0x" << std::hex << dst_addr
        << " bytes=" << std::dec << size;
    log_event(log.str());
    check(transport_.write_constants(dst_addr, constants, size));
}

/* Merge adjacent free instruction-memory blocks after evicting artifacts. */
void Context::coalesce_program_blocks() {
    for (std::size_t i = 0; i + 1 < program_blocks_.size();) {
        Block &block = program_blocks_[i];
        const Block &next = program_blocks_[i + 1];
        if (block.free && next.free && block.offset + block.size == next.offset) {
            block.size += next.size;
            program_blocks_.erase(
                program_blocks_.begin() + static_cast<std::ptrdiff_t>(i + 1));
        } else {
            ++i;
        }
    }
}

/* Evict the least-recently-used resident program artifact. */
bool Context::evict_lru_program() {
    if (program_cache_.empty()) {
        return false;
    }

    auto victim = std::min_element(
        program_cache_.begin(),
        program_cache_.end(),
        [](const ProgramCacheEntry &a, const ProgramCacheEntry &b) {
            return a.last_used < b.last_used;
        });

    std::ostringstream log;
    log << "evict_program name=" << victim->name
        << " addr=0x" << std::hex << victim->addr
        << " bytes=" << std::dec << victim->size;
    log_event(log.str());

    for (auto &block : program_blocks_) {
        if (!block.free && block.offset == victim->addr && block.size == victim->size) {
            block.free = true;
            break;
        }
    }
    program_cache_.erase(victim);
    coalesce_program_blocks();
    return true;
}

/* Allocate instruction memory, evicting old artifacts until one contiguous slot exists. */
DeviceAddress Context::allocate_program_slot(std::string_view name, std::size_t size) {
    if (size == 0 || (size % sizeof(std::uint32_t)) != 0u) {
        throw Error(Status::BadArgument);
    }
    if (size > program_memory_size_) {
        throw Error(Status::NoMemory);
    }

    while (true) {
        for (std::size_t index = 0; index < program_blocks_.size(); ++index) {
            Block &block = program_blocks_[index];
            if (!block.free || block.size < size) {
                continue;
            }

            const DeviceAddress addr = block.offset;
            if (block.size > size) {
                Block tail{
                    block.offset + static_cast<DeviceAddress>(size),
                    block.size - size,
                    true,
                };
                block.size = size;
                block.free = false;
                program_blocks_.insert(
                    program_blocks_.begin() + static_cast<std::ptrdiff_t>(index + 1),
                    tail);
            } else {
                block.free = false;
            }

            std::ostringstream log;
            log << "allocate_program name=" << name
                << " addr=0x" << std::hex << addr
                << " bytes=" << std::dec << size;
            log_event(log.str());
            return addr;
        }

        if (!evict_lru_program()) {
            throw Error(Status::NoMemory);
        }
    }
}

/* Upload a registered artifact on cache miss and return its loaded byte address. */
DeviceAddress Context::ensure_program_loaded(const PrecompiledKernel &kernel) {
    const std::string_view cache_name =
        kernel.program_id.empty() ? std::string_view(kernel.name) : std::string_view(kernel.program_id);
    for (auto &entry : program_cache_) {
        if (entry.name == cache_name) {
            entry.last_used = ++program_cache_clock_;
            return entry.addr;
        }
    }

    const void *program = kernel.program_bytes.empty()
        ? kernel.program
        : kernel.program_bytes.data();
    const std::size_t program_size = kernel.program_bytes.empty()
        ? kernel.program_size
        : kernel.program_bytes.size();
    if (!program || program_size == 0) {
        throw Error(Status::BadArgument);
    }

    const DeviceAddress addr = allocate_program_slot(cache_name, program_size);
    write_program(addr, program, program_size);
    program_cache_.push_back(ProgramCacheEntry{
        std::string(cache_name),
        addr,
        program_size,
        ++program_cache_clock_,
    });
    return addr;
}

/* Upload optional program/constants, launch the kernel, and wait if supported. */
void Context::launch_kernel(const Kernel &kernel) {
    if (kernel.program_size && !kernel.program) {
        throw Error(Status::BadArgument);
    }
    if (kernel.constants_size && !kernel.constants) {
        throw Error(Status::BadArgument);
    }

    if (kernel.program && kernel.program_size) {
        write_program(kernel.program_addr, kernel.program, kernel.program_size);
    }

    if(kernel.program_size > 0){
        check(transport_.validate_kernel(kernel.kernel_id));
    }

    if (kernel.constants && kernel.constants_size) {
        write_constants(kernel.constants_addr, kernel.constants, kernel.constants_size);
    }

    if (!transport_.launch) {
        throw Error(Status::Unsupported);
    }

    std::ostringstream log;
    log << "launch base_pc=0x" << std::hex << kernel.base_pc
        << " grid=" << std::dec << kernel.grid_dim
        << " block=" << kernel.block_dim
        << " mask=0x" << std::hex << kernel.active_mask;
    log_event(log.str());
    check(transport_.launch(
        kernel.base_pc, kernel.grid_dim, kernel.block_dim, kernel.active_mask));

    if (transport_.wait) {
        log_event("wait begin");
        check(transport_.wait(kernel.timeout_ms));
        log_event("wait done");
    }
}

/* Store a named precompiled kernel, replacing an older entry with the same name. */
void Context::register_kernel(PrecompiledKernel kernel) {
    if (!kernel.program_bytes.empty()) {
        kernel.program = nullptr;
        kernel.program_size = kernel.program_bytes.size();
    }
    if (kernel.name.empty() ||
        ((!kernel.program || kernel.program_size == 0) && kernel.program_bytes.empty())) {
        throw Error(Status::BadArgument);
    }

    for (auto &registered : kernels_) {
        if (registered.name == kernel.name) {
            registered = std::move(kernel);
            std::ostringstream log;
            log << "register_kernel name=" << registered.name
                << " bytes=" << registered.program_size
                << " entry_pc=0x" << std::hex << registered.entry_pc;
            log_event(log.str());
            return;
        }
    }

    kernels_.push_back(std::move(kernel));
    std::ostringstream log;
    log << "register_kernel name=" << kernels_.back().name
        << " bytes=" << kernels_.back().program_size
        << " entry_pc=0x" << std::hex << kernels_.back().entry_pc;
    log_event(log.str());
}

/* Load a precompiled kernel artifact and register it under a stable name. */
void Context::register_kernel_from_file(
    std::string name,
    std::string_view path,
    DeviceAddress program_addr,
    DeviceAddress base_pc,
    LaunchConfig defaults) {
    PrecompiledKernel kernel;
    kernel.name = std::move(name);
    kernel.program_id = kernel.name;
    kernel.program_bytes = read_kernel_file(path);
    kernel.program_addr = program_addr;
    kernel.base_pc = base_pc;
    kernel.entry_pc = base_pc;
    kernel.default_grid_dim = defaults.grid_dim;
    kernel.default_block_dim = defaults.block_dim;
    kernel.default_active_mask = defaults.active_mask;
    kernel.default_timeout_ms = defaults.timeout_ms;
    register_kernel(std::move(kernel));
}

/* Look up a precompiled kernel without throwing. */
const PrecompiledKernel *Context::find_kernel(std::string_view name) const noexcept {
    for (const auto &kernel : kernels_) {
        if (kernel.name == name) {
            return &kernel;
        }
    }

    return nullptr;
}

/* Build a constants table from args, upload the registered program, and launch it. */
void Context::launch_kernel(
    std::string_view name,
    const std::vector<KernelArg> &args,
    const LaunchConfig *launch_config) {
    const PrecompiledKernel *registered = find_kernel(name);
    if (!registered) {
        throw Error(Status::NotFound);
    }

    // get registered kernel ID from generated map
    auto kernel_id_it = KERNEL_ID_MAP.find(std::string(name));
    if (kernel_id_it == KERNEL_ID_MAP.end()) {
        throw Error(Status::NotFound);
    }
    uint8_t kernel_id = kernel_id_it->second;

    const LaunchConfig defaults{
        registered->default_grid_dim,
        registered->default_block_dim,
        registered->default_active_mask,
        registered->default_timeout_ms,
    };
    const LaunchConfig &launch = launch_config ? *launch_config : defaults;

    std::vector<std::uint32_t> constants;
    constants.reserve(std::max<std::size_t>(args.size(), 33u));
    for (const KernelArg &arg : args) {
        constants.push_back(arg.value);
    }
    if (constants.size() < 33u) {
        constants.resize(33u, 0u);
    }
    constants[32] = compiler_spill_base_ / 4u;

    const DeviceAddress loaded_program_addr = ensure_program_loaded(*registered);

    Kernel kernel;
    kernel.program = nullptr;
    kernel.program_size = 0;
    kernel.program_addr = loaded_program_addr;
    kernel.constants = constants.empty() ? nullptr : constants.data();
    kernel.constants_size = constants.size() * sizeof(std::uint32_t);
    kernel.constants_addr = 0;
    kernel.base_pc = (loaded_program_addr / sizeof(std::uint32_t)) + registered->entry_pc;
    kernel.grid_dim = launch.grid_dim;
    kernel.block_dim = launch.block_dim;
    kernel.active_mask = launch.active_mask;
    kernel.timeout_ms = launch.timeout_ms;
    kernel.kernel_id = kernel_id;
    std::ostringstream log;
    log << "launch_kernel name=" << name << " args=" << args.size();
    for (std::size_t i = 0; i < args.size(); ++i) {
        log << " arg" << i << "=0x" << std::hex << args[i].value;
    }
    log_event(log.str());
    launch_kernel(kernel);
}

/* Return bytes available to user allocations, excluding compiler spill scratch. */
std::size_t Context::memory_size() const noexcept {
    return memory_size_ - compiler_spill_size_;
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

/* Append a compact runtime trace line to the configured log file. */
void Context::log_event(std::string_view event) const noexcept {
    if (!logging_enabled_) {
        return;
    }

    try {
        std::ofstream log(log_path_, std::ios::app);
        if (!log) {
            return;
        }

        const auto now = std::chrono::system_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        log << ms << " " << event << "\n";
    } catch (...) {
    }
}

} // namespace minigpu
