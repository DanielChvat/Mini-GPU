#!/usr/bin/env python3
"""Extract the Clang AST for the Mini-GPU CUDA subset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CUDA_PRELUDE = """\
#ifndef MINIGPU_CUDA_PRELUDE_H
#define MINIGPU_CUDA_PRELUDE_H

#ifndef __global__
#define __global__ __attribute__((global))
#endif

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

struct dim3 {
  unsigned int x, y, z;
};

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned short half;
typedef unsigned short __half;
typedef unsigned char fp8;
typedef unsigned char fp8_e4m3;
typedef unsigned char __nv_fp8_e4m3;

extern __device__ const dim3 threadIdx;
extern __device__ const dim3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
extern __device__ void __syncthreads();

#endif
"""

DEFAULT_GPU_ARCH = "sm_50"


def detect_clang(clang_exe: str | None = None) -> str:
    """Resolve the Clang executable from an override, env, or PATH."""
    candidates = [clang_exe, os.environ.get("CLANGXX"), "clang++"]
    for candidate in candidates:
        if not candidate or candidate == "auto":
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("could not find clang++; pass --clang or set CLANGXX")


def detect_cuda_path(cuda_path: str | None = None) -> str:
    """Resolve the CUDA toolkit path from an override, env, nvcc, or common dirs."""
    candidates = [cuda_path, os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    for candidate in candidates:
        if candidate and candidate != "auto" and is_cuda_path(Path(candidate)):
            return str(Path(candidate).resolve())

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidate = Path(nvcc).resolve().parents[1]
        if is_cuda_path(candidate):
            return str(candidate)

    for candidate in (Path("/opt/cuda"), Path("/usr/local/cuda")):
        if is_cuda_path(candidate):
            return str(candidate)

    raise RuntimeError("could not find CUDA toolkit; pass --cuda-path or set CUDA_PATH")


def is_cuda_path(path: Path) -> bool:
    """Check for a CUDA toolkit directory that Clang can use."""
    return path.exists() and (path / "bin" / "nvcc").exists()


def detect_gpu_arch(gpu_arch: str | None = None) -> str:
    """Resolve the CUDA parse architecture."""
    if gpu_arch and gpu_arch != "auto":
        return gpu_arch
    return os.environ.get("MINIGPU_CUDA_ARCH", DEFAULT_GPU_ARCH)


def detect_toolchain(
    *,
    clang_exe: str | None = None,
    cuda_path: str | None = None,
    gpu_arch: str | None = None,
) -> tuple[str, str, str]:
    """Resolve all external compiler settings."""
    return (
        detect_clang(clang_exe),
        detect_cuda_path(cuda_path),
        detect_gpu_arch(gpu_arch),
    )


def parse_json_documents(text: str) -> list[dict[str, Any]]:
    """Decode Clang's one-or-more JSON AST documents."""
    decoder = json.JSONDecoder()
    documents: list[dict[str, Any]] = []
    index = 0

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break

        try:
            document, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: clang did not emit valid JSON: {exc}") from exc

        if not isinstance(document, dict):
            raise SystemExit("error: clang emitted an unexpected non-object JSON document")
        documents.append(document)

    if not documents:
        raise SystemExit("error: clang did not emit any JSON AST documents")
    return documents


def run_clang(
    source: Path,
    prelude: Path,
    *,
    clang_exe: str | None = None,
    cuda_path: str | None = None,
    gpu_arch: str | None = None,
) -> list[dict[str, Any]]:
    """Run Clang in CUDA parse mode and return raw AST documents."""
    clang, cuda_path, gpu_arch = detect_toolchain(
        clang_exe=clang_exe,
        cuda_path=cuda_path,
        gpu_arch=gpu_arch,
    )

    cmd = [
        clang,
        "-x",
        "cuda",
        f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch={gpu_arch}",
        "-nocudainc",
        "-nocudalib",
        "-include",
        str(prelude),
        "-Xclang",
        "-ast-dump=json",
        "-fsyntax-only",
        str(source),
    ]

    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise RuntimeError(f"clang AST dump failed with exit code {result.returncode}")

    if result.stderr:
        sys.stderr.write(result.stderr)

    return parse_json_documents(result.stdout)


def loc_file(node: dict[str, Any]) -> str | None:
    """Return the source file Clang attached to this AST node."""
    loc = node.get("loc", {})
    if "file" in loc:
        return loc["file"]

    begin = node.get("range", {}).get("begin", {})
    if "file" in begin:
        return begin["file"]

    expansion_loc = begin.get("expansionLoc", {})
    if "file" in expansion_loc:
        return expansion_loc["file"]

    return None


def is_from_source(node: dict[str, Any], source: Path) -> bool:
    """Check whether a node originated in the input source file."""
    file_name = loc_file(node)
    if file_name is None:
        return False
    try:
        return Path(file_name).resolve() == source.resolve()
    except OSError:
        return file_name == str(source)


def has_cuda_global_attr(node: dict[str, Any]) -> bool:
    """Detect `__global__` kernel declarations."""
    return any(child.get("kind") == "CUDAGlobalAttr" for child in node.get("inner", []))


def strip_implicit_nodes(node: Any) -> Any:
    """Remove Clang implicit nodes from compact AST output."""
    if isinstance(node, list):
        stripped = [strip_implicit_nodes(item) for item in node]
        return [item for item in stripped if item is not None]

    if not isinstance(node, dict):
        return node

    if node.get("isImplicit"):
        return None

    stripped_node: dict[str, Any] = {}
    for key, value in node.items():
        if key == "inner":
            children = strip_implicit_nodes(value)
            if children:
                stripped_node[key] = children
        else:
            stripped_node[key] = strip_implicit_nodes(value)
    return stripped_node


def find_kernels(ast: dict[str, Any], source: Path, keep_implicit: bool) -> list[dict[str, Any]]:
    """Collect source-level CUDA kernels from one translation unit."""
    kernels: list[dict[str, Any]] = []
    for node in ast.get("inner", []):
        if (
            node.get("kind") == "FunctionDecl"
            and is_from_source(node, source)
            and has_cuda_global_attr(node)
        ):
            kernels.append(node if keep_implicit else strip_implicit_nodes(node))
    return kernels


def is_device_stub(kernel: dict[str, Any]) -> bool:
    """Clang also emits host launch stubs; prefer the real device function."""
    return "__device_stub__" in kernel.get("mangledName", "")


def build_output(
    asts: list[dict[str, Any]],
    source: Path,
    *,
    full: bool = False,
    keep_implicit: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Build either the full AST dump or compact kernel-only AST JSON."""
    source = source.resolve()
    if full:
        return asts[0] if len(asts) == 1 else asts

    kernels_by_id: dict[str, dict[str, Any]] = {}
    for ast in asts:
        for kernel in find_kernels(ast, source, keep_implicit):
            loc = kernel.get("loc", {})
            kernel_id = (
                f"{kernel.get('name', '<anonymous>')}:"
                f"{loc.get('line', '?')}:"
                f"{loc.get('col', '?')}"
            )
            existing = kernels_by_id.get(kernel_id)
            if existing is not None and not is_device_stub(existing) and is_device_stub(kernel):
                continue
            kernels_by_id[kernel_id] = kernel

    kernels = list(kernels_by_id.values())
    return {
        "source": str(source),
        "translation_unit_count": len(asts),
        "kernel_count": len(kernels),
        "kernels": kernels,
    }


def get_cuda_ast(
    source: Path,
    *,
    clang_exe: str | None = None,
    cuda_path: str | None = None,
    gpu_arch: str | None = None,
    full: bool = False,
    keep_implicit: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse a CUDA-like source file and return AST JSON data."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    with tempfile.TemporaryDirectory(prefix="minigpu-ast-") as temp_dir:
        prelude = Path(temp_dir) / "cuda_prelude.h"
        prelude.write_text(CUDA_PRELUDE, encoding="utf-8")
        asts = run_clang(
            source,
            prelude,
            clang_exe=clang_exe,
            cuda_path=cuda_path,
            gpu_arch=gpu_arch,
        )

    return build_output(asts, source, full=full, keep_implicit=keep_implicit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a Mini-GPU CUDA AST.")
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--clang", default="auto", help="Clang++ executable (default: auto)")
    parser.add_argument("--cuda-path", default="auto", help="CUDA toolkit path (default: auto)")
    parser.add_argument("--gpu-arch", default="auto", help=f"CUDA parse arch (default: {DEFAULT_GPU_ARCH})")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--keep-implicit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = get_cuda_ast(
        args.source,
        clang_exe=args.clang,
        cuda_path=args.cuda_path,
        gpu_arch=args.gpu_arch,
        full=args.full,
        keep_implicit=args.keep_implicit,
    )
    encoded = json.dumps(output, indent=2)

    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
