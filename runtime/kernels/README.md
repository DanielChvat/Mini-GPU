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
the base PC and uses address 0 when uploading the artifact.

Example:

```yaml
artifacts:
  - file: vector_add.bin
    kernels:
      - name: vector_add.i32
        op: vector_add
        dtype: i32
        entry: vector_add
```
