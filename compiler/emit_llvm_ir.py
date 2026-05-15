#!/usr/bin/env python3
"""Emit LLVM IR for the Mini-GPU CUDA subset using Clang's CUDA frontend."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from get_cuda_ast import MINIGPU_KERNEL_HEADER, detect_toolchain, materialize_kernel_source


def emit_llvm_ir(
    source: Path,
    *,
    clang_exe: str | None = None,
    cuda_path: str | None = None,
    gpu_arch: str | None = None,
    kernel_dtypes: str | None = None,
) -> str:
    """Compile CUDA source to device-only textual LLVM IR."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    clang, cuda_path, gpu_arch = detect_toolchain(
        clang_exe=clang_exe,
        cuda_path=cuda_path,
        gpu_arch=gpu_arch,
    )

    with tempfile.TemporaryDirectory(prefix="minigpucc-") as tmp:
        prepared_source = materialize_kernel_source(source, kernel_dtypes, Path(tmp))
        cmd = [
            clang,
            "-x",
            "cuda",
            f"--cuda-path={cuda_path}",
            f"--cuda-gpu-arch={gpu_arch}",
            "-nocudainc",
            "-nocudalib",
            "-I",
            str(MINIGPU_KERNEL_HEADER.parent),
            "-include",
            str(MINIGPU_KERNEL_HEADER),
            "--cuda-device-only",
            "-S",
            "-emit-llvm",
            str(prepared_source),
            "-o",
            "-",
        ]

        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise RuntimeError(f"clang LLVM IR emission failed with exit code {result.returncode}")
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit device-only LLVM IR for Mini-GPU CUDA.")
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--clang", default="auto", help="Clang++ executable (default: auto)")
    parser.add_argument("--cuda-path", default="auto", help="CUDA toolkit path (default: auto)")
    parser.add_argument("--gpu-arch", default="auto", help="CUDA parse arch (default: sm_50)")
    parser.add_argument(
        "--kernel-dtypes",
        default="all",
        help="Comma-separated dtype instantiations for templated kernels, e.g. fp32,i32 (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llvm_ir = emit_llvm_ir(
        args.source,
        clang_exe=args.clang,
        cuda_path=args.cuda_path,
        gpu_arch=args.gpu_arch,
        kernel_dtypes=args.kernel_dtypes,
    )
    if args.output:
        args.output.write_text(llvm_ir, encoding="utf-8")
    else:
        print(llvm_ir, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
