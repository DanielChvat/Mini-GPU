#ifndef MINIGPU_TORCH_OPS_ELEMENTWISE_HPP
#define MINIGPU_TORCH_OPS_ELEMENTWISE_HPP

#include <ATen/ATen.h>

namespace minigpu::torch_backend::detail {

/* Launch the manifest vector-add kernel over a Mini-GPU tensor pair. */
at::Tensor run_vector_add_kernel(
    const at::Tensor &a,
    const at::Tensor &b,
    const char *op_name);

} // namespace minigpu::torch_backend::detail

#endif
