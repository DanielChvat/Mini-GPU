#include "minigpu_runtime.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

struct MockDevice {
    std::vector<std::uint8_t> memory;
    std::vector<std::uint8_t> program;
    std::vector<std::uint8_t> constants;
    bool launched = false;
    minigpu::DeviceAddress base_pc = 0;
    std::uint32_t grid_dim = 0;
    std::uint32_t block_dim = 0;
    std::uint32_t active_mask = 0;
};

static minigpu::Transport make_mock_transport(MockDevice &dev) {
    minigpu::Transport transport;
    transport.write_data =
        [&dev](minigpu::DeviceAddress dst, const void *src, std::size_t size) {
            if (dst + size > dev.memory.size()) {
                return minigpu::Status::OutOfRange;
            }
            std::memcpy(dev.memory.data() + dst, src, size);
            return minigpu::Status::Ok;
        };
    transport.read_data =
        [&dev](minigpu::DeviceAddress src, void *dst, std::size_t size) {
            if (src + size > dev.memory.size()) {
                return minigpu::Status::OutOfRange;
            }
            std::memcpy(dst, dev.memory.data() + src, size);
            return minigpu::Status::Ok;
        };
    transport.write_program =
        [&dev](minigpu::DeviceAddress dst, const void *src, std::size_t size) {
            if (dst + size > dev.program.size()) {
                dev.program.resize(dst + size);
            }
            std::memcpy(dev.program.data() + dst, src, size);
            return minigpu::Status::Ok;
        };
    transport.write_constants =
        [&dev](minigpu::DeviceAddress dst, const void *src, std::size_t size) {
            if (dst + size > dev.constants.size()) {
                dev.constants.resize(dst + size);
            }
            std::memcpy(dev.constants.data() + dst, src, size);
            return minigpu::Status::Ok;
        };
    transport.launch =
        [&dev](
            minigpu::DeviceAddress base_pc, std::uint32_t grid_dim,
            std::uint32_t block_dim, std::uint32_t active_mask) {
            dev.launched = true;
            dev.base_pc = base_pc;
            dev.grid_dim = grid_dim;
            dev.block_dim = block_dim;
            dev.active_mask = active_mask;
            return minigpu::Status::Ok;
        };
    transport.wait = [](std::uint32_t) {
        return minigpu::Status::Ok;
    };

    return transport;
}

int main() {
    MockDevice dev;
    dev.memory.resize(1024);

    minigpu::Config config;
    config.memory_size = dev.memory.size();
    config.default_alignment = 16;
    config.transport = make_mock_transport(dev);

    minigpu::Context context(std::move(config));
    assert(context.memory_size() == 1024);

    auto a = context.device_malloc(24);
    auto b = context.device_malloc_aligned(32, 32);
    assert(a.addr() % 16 == 0);
    assert(b.addr() % 32 == 0);

    const std::uint32_t input[4] = {1, 2, 3, 4};
    std::uint32_t output[4] = {};
    context.copy_to_device(a.addr(), input, sizeof(input));
    context.copy_from_device(output, a.addr(), sizeof(output));
    assert(std::memcmp(input, output, sizeof(input)) == 0);

    const std::uint32_t program[2] = {0x12345678u, 0x9abcdef0u};
    const std::uint32_t constants[1] = {0x40e00000u};
    minigpu::Kernel kernel;
    kernel.program = program;
    kernel.program_size = sizeof(program);
    kernel.constants = constants;
    kernel.constants_size = sizeof(constants);
    kernel.grid_dim = 1;
    kernel.block_dim = 4;
    kernel.active_mask = 0xf;
    kernel.timeout_ms = 1000;

    context.launch_kernel(kernel);
    assert(dev.launched);
    assert(dev.grid_dim == 1);
    assert(dev.block_dim == 4);
    assert(dev.active_mask == 0xf);
    assert(dev.program.size() == sizeof(program));
    assert(dev.constants.size() == sizeof(constants));

    a.reset();
    b.reset();
    assert(context.memory_free() == context.memory_size());
}
