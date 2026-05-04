#ifndef MINIGPU_TORCH_HPP
#define MINIGPU_TORCH_HPP

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include "minigpu_runtime.hpp"

namespace minigpu::torch_backend {

/* Initialize global Mini-GPU/PyTorch backend state. */
void init();

/* Return true when the extension was compiled into this Python package. */
bool is_built();

/* Return true when at least one Mini-GPU device can be opened. */
bool is_available();

/* Return the number of Mini-GPU devices visible to PyTorch. */
int device_count();

/* Return the active Mini-GPU device index for the current process/thread. */
int get_device();

/* Set the active Mini-GPU device index for later runtime calls. */
void set_device(int index);

/* Return the active Mini-GPU runtime context. */
minigpu::Context &runtime_context();

/* Convert communication errors into runtime status values. */
minigpu::Status map_com_status(int err);

/* Build a runtime transport for a communication device. */
minigpu::Transport make_gpu_comm_transport(void *dev);

/* Allocate a PrivateUse1 tensor. */
at::Tensor empty(
    at::IntArrayRef size,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format);

/* Copy data between CPU and Mini-GPU tensors. */
at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking);

/* Elementwise add operation for Mini-GPU tensors. */
at::Tensor add_tensor(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha);

/* Two-input custom vector-add op wrapper for torch.ops.minigpu.vector_add. */
at::Tensor vector_add(const at::Tensor &a, const at::Tensor &b);

/* Elementwise multiply operation for Mini-GPU tensors. */
at::Tensor mul_tensor(const at::Tensor &a, const at::Tensor &b);

/* Matrix multiply operation for Mini-GPU tensors. */
at::Tensor mm(const at::Tensor &a, const at::Tensor &b);

/* ReLU operation for Mini-GPU tensors. */
at::Tensor relu(const at::Tensor &a);

} // namespace minigpu::torch_backend

#endif
