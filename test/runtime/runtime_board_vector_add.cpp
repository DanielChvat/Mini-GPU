#include "minigpu_buffer.hpp"
#include "minigpu_kernels.hpp"
#include "transports/gpu_comm_transport.hpp"

#include <cstdio>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s /dev/ttyUSB1\n", argv[0]);
        return 1;
    }

    const char *port = argv[1];

    try {
        com_dev_t *dev = open_com(port, 115200, 5000);
        if (!dev) {
            std::fprintf(stderr, "failed to open %s\n", port);
            return 1;
        }

        minigpu::Config config;
        config.memory_base = 0;
        config.memory_size = 32u * 1024u;
        config.default_alignment = 4;
        config.transport = minigpu::transports::make_gpu_comm_transport(dev);

        minigpu::Context context(std::move(config));
        minigpu::kernels::register_builtin_kernels(context);

        auto a = minigpu::Tensor<std::uint32_t>(context, 4);
        auto b = minigpu::Tensor<std::uint32_t>(context, 4);
        auto c = minigpu::Tensor<std::uint32_t>(context, 4);

        const std::vector<std::uint32_t> host_a = {100, 101, 102, 103};
        const std::vector<std::uint32_t> host_b = {7, 10, 13, 16};
        const std::vector<std::uint32_t> expected = {107, 111, 115, 119};

        std::printf("A addr=0x%08x B addr=0x%08x C addr=0x%08x\n",
                    a.addr(), b.addr(), c.addr());
        std::printf("copying inputs...\n");
        a.write_all(host_a);
        b.write_all(host_b);

        std::printf("launching vector_add_i32...\n");
        context.launch_kernel(
            "vector_add_i32",
            {
                minigpu::KernelArg::device_ptr(a.addr()),
                minigpu::KernelArg::device_ptr(b.addr()),
                minigpu::KernelArg::device_ptr(c.addr()),
            });

        std::printf("reading output...\n");
        std::vector<std::uint32_t> out = c.read_all();
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::printf("C[%zu] = %u\n", i, out[i]);
        }

        close_com(dev);
        if (out != expected) {
            std::fprintf(stderr, "vector_add_i32 mismatch\n");
            return 1;
        }

        std::printf("runtime vector_add_i32 passed\n");
        return 0;
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "%s\n", exc.what());
        return 1;
    }
}
