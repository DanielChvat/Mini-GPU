# Mini-GPU Kernel Artifacts

Put precompiled Mini-GPU kernel artifacts in this folder.

The runtime loader supports:

- `.bin`: raw instruction bytes
- `.hex`: newline-separated 32-bit instruction words

Repository kernels should use `.bin` artifacts. Save IR/ISA/hex debug output
outside this folder unless you intentionally want to check it in.

Set `MINIGPU_KERNEL_DIR=/path/to/kernels` to load kernels from another folder.

Kernel metadata lives in `kernels.yaml`. Add a new kernel by adding a manifest
entry and placing the compiled artifact beside it.

Example:

```yaml
kernels:
  - name: vector_add.i32
    op: vector_add
    dtype: i32
    file: vector_add.bin
```
