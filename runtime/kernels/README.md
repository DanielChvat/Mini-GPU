# Mini-GPU Kernel Artifacts

Put precompiled Mini-GPU kernel artifacts in this folder.

The runtime loader supports:

- `.bin`: raw instruction bytes
- `.map`: compiler-generated kernel entry metadata

Repository kernels should use `.bin` plus `.map` artifacts. Save IR/ISA/hex
debug output outside this folder unless you intentionally want to check it in.

Set `MINIGPU_KERNEL_DIR=/path/to/kernels` to load kernels from another folder.

Kernel metadata lives in `kernels.yaml`. Add kernels under an artifact and point
each entry at the CUDA function name. The runtime reads the `.map` file to find
the artifact-local PC. An artifact can set `program_addr` to place its
instruction words at a byte address in program BRAM; kernel launch PCs are
computed as `program_addr / 4 + entry_pc`.

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

```yaml
artifacts:
  - file: vector_add.bin
    program_addr: 0x0000
    kernels:
      - name: vector_add.i32
        op: vector_add
        dtype: i32
        entry: vector_add
```
