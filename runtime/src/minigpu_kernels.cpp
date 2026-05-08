#include "minigpu_kernels.hpp"

#include <cstdlib>
#include <string>

#ifndef MINIGPU_DEFAULT_KERNEL_DIR
#define MINIGPU_DEFAULT_KERNEL_DIR "runtime/kernels"
#endif

namespace minigpu::kernels {

/* Return the kernel artifact folder, allowing an environment override. */
std::string default_kernel_dir() {
    const char *override_dir = std::getenv("MINIGPU_KERNEL_DIR");
    if (override_dir && override_dir[0] != '\0') {
        return override_dir;
    }

    return MINIGPU_DEFAULT_KERNEL_DIR;
}

/* Register the int32 vector-add kernel from the configured kernel folder. */
void register_vector_add_i32(Context &context) {
    LaunchConfig defaults;
    defaults.grid_dim = 1;
    defaults.block_dim = 4;
    defaults.active_mask = 0x0000000fu;
    defaults.timeout_ms = 5000;

    context.register_kernel_from_file(
        "vector_add_i32",
        default_kernel_dir() + "/vector_add_i32.hex",
        0,
        0,
        defaults);
}

/* Register all built-in kernels with the provided context. */
void register_builtin_kernels(Context &context) {
    register_vector_add_i32(context);
}

} // namespace minigpu::kernels
