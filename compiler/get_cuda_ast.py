#!/usr/bin/env python3
"""Extract the Clang AST for the Mini-GPU CUDA subset."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_GPU_ARCH = "sm_50"
REPO_ROOT = Path(__file__).resolve().parents[1]
MINIGPU_KERNEL_HEADER = REPO_ROOT / "runtime" / "kernels" / "include" / "minigpu_kernel.h"
KERNEL_DTYPE_TYPES = {
    "i32": "int",
    "i16": "int16_t",
    "i8": "int8_t",
    "fp32": "float",
    "fp16": "half",
    "fp8": "fp8_e4m3fn",
}
TEMPLATE_KERNEL_RE = re.compile(
    r"template\s*<\s*(?:typename|class)\s+([A-Za-z_]\w*)\s*>\s*"
    r"__global__\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)",
    re.DOTALL,
)
EXPLICIT_TEMPLATE_RE = re.compile(
    r"template\s+__global__\s+void\s+([A-Za-z_]\w*)\s*<",
    re.DOTALL,
)


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


def kernel_dtypes(dtype_text: str | None = None) -> list[str]:
    """Return the dtype suffixes selected for templated kernel instantiation."""
    if not dtype_text or dtype_text == "all":
        return list(KERNEL_DTYPE_TYPES)

    requested = {part.strip().lower() for part in dtype_text.split(",") if part.strip()}
    if "all" in requested:
        return list(KERNEL_DTYPE_TYPES)

    unknown = sorted(requested - set(KERNEL_DTYPE_TYPES))
    if unknown:
        raise RuntimeError(f"unknown Mini-GPU kernel dtype(s): {', '.join(unknown)}")

    return [dtype for dtype in KERNEL_DTYPE_TYPES if dtype in requested]


def split_parameters(params: str) -> list[str]:
    """Split a simple C parameter list while respecting template angle brackets."""
    if not params.strip() or params.strip() == "void":
        return []

    split: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(params):
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            split.append(params[start:index].strip())
            start = index + 1
    split.append(params[start:].strip())
    return [param for param in split if param]


def parameter_type(param: str) -> str:
    """Strip a parameter name, leaving the type spelling for explicit instantiation."""
    clean = re.sub(r"\s*=\s*.*$", "", param.strip())
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"([*&])\s*[A-Za-z_]\w*$", r"\1", clean).strip()
    clean = re.sub(r"\s+[A-Za-z_]\w*$", "", clean).strip()
    return clean


def instantiate_template_kernel(
    *,
    template_param: str,
    kernel_name: str,
    params: str,
    dtype: str,
) -> str:
    """Build one explicit `template __global__` instantiation."""
    concrete_type = KERNEL_DTYPE_TYPES[dtype]
    param_types = []
    for param in split_parameters(params):
        param_types.append(
            re.sub(rf"\b{re.escape(template_param)}\b", concrete_type, parameter_type(param))
        )
    return f"template __global__ void {kernel_name}<{concrete_type}>({', '.join(param_types)});"


def template_instantiation_footer(source_text: str, dtype_text: str | None = None) -> str:
    """Generate explicit instantiations for source-level templated kernels."""
    explicit = set(EXPLICIT_TEMPLATE_RE.findall(source_text))
    lines: list[str] = []
    for match in TEMPLATE_KERNEL_RE.finditer(source_text):
        template_param, kernel_name, params = match.groups()
        if kernel_name in explicit:
            continue
        for dtype in kernel_dtypes(dtype_text):
            lines.append(
                instantiate_template_kernel(
                    template_param=template_param,
                    kernel_name=kernel_name,
                    params=params,
                    dtype=dtype,
                )
            )
    if not lines:
        return ""
    return "\n\n// Generated by minigpucc from --kernel-dtypes.\n" + "\n".join(lines) + "\n"


def materialize_kernel_source(source: Path, dtype_text: str | None, temp_dir: Path) -> Path:
    """Write the compiler-facing source with generated template instantiations."""
    source_text = source.read_text(encoding="utf-8")
    footer = template_instantiation_footer(source_text, dtype_text)
    if not footer:
        return source

    prepared = temp_dir / source.name
    prepared.write_text(source_text.rstrip() + footer, encoding="utf-8")
    return prepared


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
    prelude: Path = MINIGPU_KERNEL_HEADER,
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
        "-I",
        str(prelude.parent),
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


def type_text(node: dict[str, Any]) -> str:
    """Return Clang's textual qualified type for a node."""
    type_info = node.get("type")
    if isinstance(type_info, dict):
        return str(type_info.get("qualType", ""))
    return ""


def format_from_qual_type(text: str) -> str | None:
    """Map a C/CUDA type spelling to a Mini-GPU dtype suffix."""
    clean = text.replace("const ", "").replace("*", "").strip()
    aliases = {
        "int": "i32",
        "int32_t": "i32",
        "short": "i16",
        "int16_t": "i16",
        "unsigned short": "fp16",
        "signed char": "i8",
        "int8_t": "i8",
        "unsigned char": "fp8",
        "float": "fp32",
        "half": "fp16",
        "__half": "fp16",
        "fp8": "fp8",
        "fp8_e4m3": "fp8",
        "fp8_e4m3fn": "fp8",
        "__nv_fp8_e4m3": "fp8",
    }
    return aliases.get(clean)


def template_suffix(node: dict[str, Any]) -> str:
    """Return a stable dtype suffix for an explicitly instantiated kernel template."""
    suffixes: list[str] = []
    for child in node.get("inner", []):
        if child.get("kind") != "TemplateArgument":
            continue
        fmt = format_from_qual_type(type_text(child))
        if fmt:
            suffixes.append(fmt)
    return "_" + "_".join(suffixes) if suffixes else ""


def kernel_export_name(node: dict[str, Any]) -> str:
    """Return the manifest-visible kernel name."""
    return str(node.get("name", "<anonymous>")) + template_suffix(node)


def walk_nodes(node: Any):
    """Yield every dictionary node in a Clang AST tree."""
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("inner", []):
        yield from walk_nodes(child)


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
    for node in walk_nodes(ast):
        if (
            node.get("kind") == "FunctionDecl"
            and is_from_source(node, source)
            and has_cuda_global_attr(node)
            and not is_device_stub(node)
            and node.get("mangledName")
        ):
            kernel = node if keep_implicit else strip_implicit_nodes(node)
            kernel["minigpuKernelName"] = kernel_export_name(node)
            kernels.append(kernel)
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
                f"{kernel.get('minigpuKernelName', kernel.get('name', '<anonymous>'))}:"
                f"{loc.get('line', '?')}:"
                f"{loc.get('col', '?')}"
                f":{kernel.get('mangledName', '')}"
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
    kernel_dtypes: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse a CUDA-like source file and return AST JSON data."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    with tempfile.TemporaryDirectory(prefix="minigpucc-") as tmp:
        prepared_source = materialize_kernel_source(source, kernel_dtypes, Path(tmp))
        asts = run_clang(
            prepared_source,
            MINIGPU_KERNEL_HEADER,
            clang_exe=clang_exe,
            cuda_path=cuda_path,
            gpu_arch=gpu_arch,
        )

        return build_output(asts, prepared_source, full=full, keep_implicit=keep_implicit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a Mini-GPU CUDA AST.")
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--clang", default="auto", help="Clang++ executable (default: auto)")
    parser.add_argument("--cuda-path", default="auto", help="CUDA toolkit path (default: auto)")
    parser.add_argument("--gpu-arch", default="auto", help=f"CUDA parse arch (default: {DEFAULT_GPU_ARCH})")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--keep-implicit", action="store_true")
    parser.add_argument(
        "--kernel-dtypes",
        default="all",
        help="Comma-separated dtype instantiations for templated kernels, e.g. fp32,i32 (default: all)",
    )
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
        kernel_dtypes=args.kernel_dtypes,
    )
    encoded = json.dumps(output, indent=2)

    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
