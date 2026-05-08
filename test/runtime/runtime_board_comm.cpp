#include "minigpu_buffer.hpp"
#include "transports/gpu_comm_transport.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s /dev/ttyUSB1 [bytes]\n", argv[0]);
        return 1;
    }

    const char *port = argv[1];
    const std::size_t byte_count = argc >= 3
        ? static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 0))
        : 300u;
    if (byte_count == 0) {
        std::fprintf(stderr, "byte count must be non-zero\n");
        return 1;
    }

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
        std::printf("opened %s and created runtime context\n", port);
        std::fflush(stdout);

        auto bytes = minigpu::Tensor<std::uint8_t>(context, byte_count);
        std::printf("allocated byte buffer: addr=0x%08x bytes=%zu\n",
                    bytes.addr(), bytes.bytes());
        std::fflush(stdout);

        std::vector<std::uint8_t> input(byte_count);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<std::uint8_t>(i & 0xffu);
        }

        std::printf("writing byte pattern...\n");
        std::fflush(stdout);
        bytes.write_all(input);
        std::printf("reading byte pattern...\n");
        std::fflush(stdout);
        std::vector<std::uint8_t> output = bytes.read_all();
        std::printf("read byte pattern\n");
        std::fflush(stdout);

        if (output != input) {
            close_com(dev);
            std::fprintf(stderr, "runtime board comm mismatch\n");
            return 1;
        }

        auto words = minigpu::Tensor<std::uint32_t>(context, 4);
        std::printf("allocated word buffer: addr=0x%08x bytes=%zu\n",
                    words.addr(), words.bytes());
        std::fflush(stdout);

        std::printf("writing word proxies...\n");
        std::fflush(stdout);
        words[0] = 0x11223344u;
        words[1] = 0x55667788u;
        words[2] = 0x99aabbccu;
        words[3] = 0xddeeff00u;

        std::printf("reading word proxies...\n");
        std::fflush(stdout);
        if (static_cast<std::uint32_t>(words[0]) != 0x11223344u ||
            static_cast<std::uint32_t>(words[1]) != 0x55667788u ||
            static_cast<std::uint32_t>(words[2]) != 0x99aabbccu ||
            static_cast<std::uint32_t>(words[3]) != 0xddeeff00u) {
            close_com(dev);
            std::fprintf(stderr, "runtime board word proxy mismatch\n");
            return 1;
        }

        std::printf("runtime board comm passed: addr=0x%08x bytes=%zu\n",
                    bytes.addr(), bytes.bytes());
        close_com(dev);
        return 0;
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "%s\n", exc.what());
        return 1;
    }
}
