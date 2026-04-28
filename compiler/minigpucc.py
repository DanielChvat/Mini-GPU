#!/usr/bin/env python3
"""Mini-GPU compiler driver: CUDA subset -> AST -> IR -> ISA assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ast_to_ir import ast_to_ir
from get_cuda_ast import get_cuda_ast
from ir_to_isa import ir_to_isa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Mini-GPU CUDA sources.")
    parser.add_argument("source", type=Path, help="CUDA-like .cu source file")
    parser.add_argument("-o", "--output", type=Path, help="Primary output file")
    parser.add_argument(
        "--emit",
        choices=("ast", "ir", "isa"),
        default="isa",
        help="Primary output stage (default: isa)",
    )
    parser.add_argument(
        "--save-ast",
        nargs="?",
        const=True,
        default=False,
        metavar="PATH",
        help="Also write compact AST JSON; default path is <source>.ast.json",
    )
    parser.add_argument(
        "--save-ir",
        nargs="?",
        const=True,
        default=False,
        metavar="PATH",
        help="Also write Mini-GPU IR; default path is <source>.ir",
    )
    parser.add_argument(
        "--save-isa",
        nargs="?",
        const=True,
        default=False,
        metavar="PATH",
        help="Also write ISA assembly; default path is <source>.isa",
    )
    parser.add_argument("--clang", default="auto", help="Clang++ executable (default: auto)")
    parser.add_argument("--cuda-path", default="auto", help="CUDA toolkit path (default: auto)")
    parser.add_argument("--gpu-arch", default="auto", help="CUDA parse arch (default: sm_50)")
    parser.add_argument("--full-ast", action="store_true", help="Emit full Clang AST for --emit ast")
    parser.add_argument("--keep-implicit", action="store_true", help="Keep implicit Clang AST nodes")
    return parser.parse_args()


def default_stage_path(source: Path, suffix: str) -> Path:
    """Build the default side-output path for one compiler stage."""
    return source.with_suffix(suffix)


def optional_path(value: bool | str, source: Path, suffix: str) -> Path | None:
    """Resolve optional `--save-*` arguments."""
    if value is False:
        return None
    if value is True:
        return default_stage_path(source, suffix)
    return Path(value)


def write_text(path: Path, text: str) -> None:
    """Write compiler output with a trailing newline."""
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def encode_ast(ast: Any) -> str:
    """Serialize AST JSON data for files or stdout."""
    return json.dumps(ast, indent=2)


def compile_source(args: argparse.Namespace) -> tuple[str, dict[str, str]]:
    """Run the requested compiler pipeline and return primary plus side outputs."""
    compact_ast = get_cuda_ast(
        args.source,
        clang_exe=args.clang,
        cuda_path=args.cuda_path,
        gpu_arch=args.gpu_arch,
        full=False,
        keep_implicit=args.keep_implicit,
    )

    ast_for_output = compact_ast
    if args.emit == "ast" and args.full_ast:
        ast_for_output = get_cuda_ast(
            args.source,
            clang_exe=args.clang,
            cuda_path=args.cuda_path,
            gpu_arch=args.gpu_arch,
            full=True,
            keep_implicit=args.keep_implicit,
        )

    ir = ast_to_ir(compact_ast)
    isa = ir_to_isa(ir)

    side_outputs = {
        "ast": encode_ast(compact_ast),
        "ir": ir,
        "isa": isa,
    }

    primary = {
        "ast": encode_ast(ast_for_output),
        "ir": ir,
        "isa": isa,
    }[args.emit]
    return primary, side_outputs


def main() -> int:
    args = parse_args()
    primary, side_outputs = compile_source(args)

    save_paths = {
        "ast": optional_path(args.save_ast, args.source, ".ast.json"),
        "ir": optional_path(args.save_ir, args.source, ".ir"),
        "isa": optional_path(args.save_isa, args.source, ".isa"),
    }
    for stage, path in save_paths.items():
        if path is not None:
            write_text(path, side_outputs[stage])

    if args.output:
        write_text(args.output, primary)
    else:
        print(primary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
