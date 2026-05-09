#ifndef MINIGPU_KERNELS_HPP
#define MINIGPU_KERNELS_HPP

#include "minigpu_runtime.hpp"

#include <string>

namespace minigpu::kernels {

/* Return the folder used for precompiled Mini-GPU kernel artifacts. */
std::string default_kernel_dir();

/* Register the built-in precompiled kernels with a runtime context. */
void register_builtin_kernels(Context &context);

/* Register the int32 vector-add kernel from a precompiled artifact file. */
void register_vector_add_i32(Context &context);

} // namespace minigpu::kernels

#endif
