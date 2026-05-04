#!/usr/bin/env python3
"""Assemble a Mini-GPU program into a testbench $readmemh file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "compiler"))

from isa_to_bin import isa_to_words  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Mini-GPU testbench hex from assembly.")
    parser.add_argument("asm", type=Path, help="Input Mini-GPU assembly file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output hex file")
    parser.add_argument(
        "--depth",
        type=int,
        default=256,
        help="Maximum instruction words supported by mini_gpu_program_tb (default: 256)",
    )
    parser.add_argument(
        "--no-pad-exit",
        action="store_true",
        help="Do not pad the output to --depth with EXIT instructions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    words = isa_to_words(args.asm.read_text(encoding="utf-8"))
    if len(words) > args.depth:
        raise SystemExit(f"error: program has {len(words)} words, but depth is {args.depth}")

    if not args.no_pad_exit:
        exit_word = 0xF0000000
        words = words + [exit_word] * (args.depth - len(words))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{word:08X}\n" for word in words), encoding="utf-8")
    print(f"wrote {len(words)} words to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
