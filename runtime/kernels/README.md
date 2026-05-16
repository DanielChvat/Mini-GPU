# Mini-GPU Kernel Artifacts

Put precompiled Mini-GPU kernel artifacts in this folder.

The runtime loader supports:

- `bin/*.bin`: raw instruction bytes
- `maps/*.map`: compiler-generated kernel entry metadata

Repository kernels should use generated `bin/` plus `maps/` artifacts. Save
IR/ISA/hex debug output outside this folder unless you intentionally want to
check it in.

Set `MINIGPU_KERNEL_DIR=/path/to/kernels` to load kernels from another folder.

Kernel metadata is generated from `MINIGPU_REGISTER_KERNEL` declarations in the
`.cu` files. Single-entry artifacts compiled with `minigpucc --only-kernel` do
not need per-symbol PCs in their `.map`; the runtime launches them at the start
of the loaded program. Legacy multi-entry artifacts may still include `pc`
values so the runtime can jump to an entry inside one instruction stream.

Templated elementwise kernels should declare the template only:

```cpp
template <typename T>
__global__ void my_kernel(T *a, T *out, int n) { /* ... */ }
```

The compiler defaults to all Mini-GPU dtypes. To build a capability-specific
artifact, pass `--kernel-dtypes fp32,i32` or another comma-separated subset.
`minigpucc` appends explicit template instantiations for the selected dtypes
before passing the source to Clang.

Example:

```cpp
template <typename T>
__global__ void my_kernel(T *a, T *out, int n) { /* ... */ }

MINIGPU_REGISTER_KERNEL("my_op.fp32", "my_op", "fp32", "my_kernel_fp32")
```

For templated kernels, register all supported dtype instantiations on one line:

```cpp
template <typename T>
__global__ void my_kernel(T *a, T *out, int n) { /* ... */ }

MINIGPU_REGISTER_TYPED_KERNELS("my_op", "i32,fp32,fp16", "my_kernel")
```

Regenerate all artifacts and `kernels.yaml` with:

```bash
python compiler/build_kernel_artifacts.py
```
