# Mini-GPU PyTorch Backend Plan

## Current Status

- [x] C++ runtime library exists.
- [x] Python package entry point exists in `torch_mini_gpu/`.
- [x] `PrivateUse1` backend is renamed to `minigpu`.
- [x] C++ PyTorch extension target exists behind `MINIGPU_BUILD_TORCH`.
- [x] PyTorch extension builds as `torch_mini_gpu/_C*.so`.
- [x] Stub backend/device helpers exist.
- [x] Stub ATen registrations exist for first operators.
- [ ] Real Mini-GPU device/context initialization works.
- [ ] Mini-GPU tensor storage/allocation works.
- [ ] CPU-to-device and device-to-CPU tensor copies work.
- [ ] PyTorch ops launch real Mini-GPU kernels.

## Runtime And Communication

- [x] `minigpu::Context`
- [x] `minigpu::DeviceBuffer`
- [x] `minigpu::Kernel`
- [x] `minigpu::Transport`
- [x] `com_write_data`
- [x] `com_read_data`
- [ ] `com_write_program`
- [ ] `com_write_constants` or documented constant-memory mapping
- [ ] `com_launch`
- [ ] `com_read_status`
- [ ] `com_wait`
- [ ] Runtime-to-`gpu_comm` transport factory

## PyTorch Backend Functions

- [x] `init`
- [x] `is_built`
- [x] `is_available`
- [x] `device_count`
- [x] `get_device`
- [x] `set_device`
- [x] `aten::empty.memory_format` stub
- [x] `aten::copy_` stub
- [x] `aten::add.Tensor` stub
- [x] `aten::mul.Tensor` stub
- [x] `aten::relu` stub
- [x] `aten::mm` stub
- [x] `minigpu::vector_add` stub
- [x] `minigpu::matmul` stub
- [x] `minigpu::relu` stub
- [ ] `aten::empty.memory_format` real implementation
- [ ] `aten::copy_` real implementation
- [ ] `aten::add.Tensor` real implementation
- [ ] `aten::mul.Tensor` real implementation
- [ ] `aten::relu` real implementation
- [ ] `aten::mm` real implementation
- [ ] Custom Mini-GPU ops real implementations

## First Operator Milestones

- [ ] Create an empty Mini-GPU tensor.
- [ ] Copy one FP32 tensor from CPU to Mini-GPU.
- [ ] Copy one FP32 tensor from Mini-GPU to CPU.
- [ ] Run FP32 vector add.
- [ ] Run FP32 multiply.
- [ ] Run ReLU.
- [ ] Run tiny matrix multiply.
- [ ] Verify results against CPU PyTorch tensors.

## Future Plans

- [ ] Add a real allocator-backed PrivateUse1 tensor storage path.
- [ ] Cache compiled/uploaded Mini-GPU programs.
- [ ] Add dtype checks for FP32 first.
- [ ] Add FP16 support after hardware/runtime support is stable.
- [ ] Add shape-specialized kernels for small neural-network demos.
- [ ] Add pytest coverage for backend import, tensor copies, and first ops.
- [ ] Package the extension for editable local install.
