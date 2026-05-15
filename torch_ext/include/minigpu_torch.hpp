#ifndef MINIGPU_TORCH_HPP
#define MINIGPU_TORCH_HPP

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include "minigpu_runtime.hpp"

#include <cstdint>
#include <string>

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

/* Open a gpu_comm serial device and create the runtime context. */
void connect(const std::string &port, std::uint32_t baud, std::uint32_t memory_size);

/* Close the gpu_comm serial device and destroy the runtime context. */
void disconnect();

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

/* Allocate a strided PrivateUse1 tensor. */
at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory);

/* Return a reshaped alias of a Mini-GPU tensor. */
at::Tensor view(const at::Tensor &self, c10::SymIntArrayRef size);

/* Return an alias with explicit size, stride, and storage offset metadata. */
at::Tensor as_strided(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    ::std::optional<c10::SymInt> storage_offset);

/* Return a reshape alias when PyTorch has already computed the target strides. */
at::Tensor reshape_alias(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride);

/* Copy data between CPU and Mini-GPU tensors. */
at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking);

/* Materialize a dtype/device converted copy for Tensor.to(), .float(), .half(), etc. */
at::Tensor to_copy(
    const at::Tensor &self,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> memory_format);

/* Return the runtime device address backing a Mini-GPU tensor. */
minigpu::DeviceAddress device_address(const at::Tensor &tensor);

/* Elementwise add operation for Mini-GPU tensors. */
at::Tensor add_tensor(const at::Tensor &a, const at::Tensor &b, const at::Scalar &alpha);

/* Two-input custom vector-add op wrapper for torch.ops.minigpu.vector_add. */
at::Tensor vector_add(const at::Tensor &a, const at::Tensor &b);

/* Elementwise multiply operation for Mini-GPU tensors. */
at::Tensor mul_tensor(const at::Tensor &a, const at::Tensor &b);

/* Elementwise division operation for Mini-GPU tensors. */
at::Tensor div_tensor(const at::Tensor &a, const at::Tensor &b);

/* Matrix multiply operation for Mini-GPU tensors. */
at::Tensor mm(const at::Tensor &a, const at::Tensor &b);

/* Linear layer operation for Mini-GPU tensors. */
at::Tensor linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const ::std::optional<at::Tensor> &bias);

/* ReLU operation for Mini-GPU tensors. */
at::Tensor relu(const at::Tensor &a);

/* Natural exponential operation for Mini-GPU tensors. */
at::Tensor exp(const at::Tensor &a);

/* Natural logarithm operation for Mini-GPU tensors. */
at::Tensor log(const at::Tensor &a);

/* Base-2 logarithm operation for Mini-GPU tensors. */
at::Tensor log2(const at::Tensor &a);

/* Base-10 logarithm operation for Mini-GPU tensors. */
at::Tensor log10(const at::Tensor &a);

/* Square-root operation for Mini-GPU tensors. */
at::Tensor sqrt(const at::Tensor &a);

/* Reciprocal operation for Mini-GPU tensors. */
at::Tensor reciprocal(const at::Tensor &a);

/* Tensor exponentiation operation for Mini-GPU tensors. */
at::Tensor pow_tensor_tensor(const at::Tensor &a, const at::Tensor &b);

/* Scalar exponentiation operation for Mini-GPU tensors. */
at::Tensor pow_tensor_scalar(const at::Tensor &a, const at::Scalar &b);

/* Sigmoid operation for Mini-GPU tensors. */
at::Tensor sigmoid(const at::Tensor &a);

/* Tanh operation for Mini-GPU tensors. */
at::Tensor tanh(const at::Tensor &a);

/* Trigonometric operations for Mini-GPU tensors. */
at::Tensor sin(const at::Tensor &a);
at::Tensor cos(const at::Tensor &a);
at::Tensor tan(const at::Tensor &a);

} // namespace minigpu::torch_backend

#endif
