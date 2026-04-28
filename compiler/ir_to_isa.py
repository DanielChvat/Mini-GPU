#!/usr/bin/env python3
"""Lower Mini-GPU IR text into Mini-GPU ISA assembly."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


IR_ASSIGN_RE = re.compile(r"^(%[\w.]+)\s*=\s*(\w+)(?:\s+(.*))?$")

THREAD_OPS = {
    "global_tid": "TID",
    "thread_idx": "TIDX",
    "block_idx": "BID",
    "block_dim": "BDIM",
    "grid_dim": "GDIM",
}

VALUE_OPS = {
    "add": "ADD",
    "sub": "SUB",
    "mul": "MUL",
    "and": "AND",
    "or": "OR",
    "xor": "XOR",
    "shl": "SHL",
    "shr": "SHR",
    "lt": "SLT",
    "le": "SLE",
    "gt": "SGT",
    "ge": "SGE",
    "eq": "SEQ",
    "ne": "SNE",
}

EXTENSION_VALUE_OPS = {
    "div": "DIV",
    "mod": "MOD",
}


class IsaLoweringError(Exception):
    """Raised when IR cannot be represented in the current ISA assembly."""

    pass


@dataclass
class RegisterAllocator:
    remaining_uses: Counter[str]
    value_regs: dict[str, str] = field(default_factory=dict)
    free_regs: list[str] = field(default_factory=lambda: [f"r{i}" for i in range(16)])

    def define(self, value: str) -> str:
        """Assign a physical register to a newly defined IR value."""
        old_reg = self.value_regs.pop(value, None)
        if old_reg is not None:
            self.free_regs.insert(0, old_reg)

        if not self.free_regs:
            raise IsaLoweringError(
                "ran out of physical registers while lowering IR; "
                "the next step is spill support or better liveness analysis"
            )

        reg = self.free_regs.pop(0)
        self.value_regs[value] = reg
        return reg

    def use(self, value: str) -> str:
        """Return the register containing an existing IR value."""
        if value not in self.value_regs:
            raise IsaLoweringError(f"use of undefined IR value: {value}")
        return self.value_regs[value]

    def release_after_use(self, value: str) -> None:
        """Free temporary registers after their final textual use."""
        if not value.startswith("%t"):
            return

        self.remaining_uses[value] -= 1
        if self.remaining_uses[value] > 0:
            return

        reg = self.value_regs.pop(value, None)
        if reg is not None:
            self.free_regs.insert(0, reg)

    def acquire_scratch(self) -> str:
        """Reserve a short-lived register for materialized immediates."""
        if not self.free_regs:
            raise IsaLoweringError("ran out of physical registers for immediate materialization")
        return self.free_regs.pop(0)

    def release_scratch(self, reg: str) -> None:
        """Return a scratch register to the free list."""
        self.free_regs.insert(0, reg)


def split_operands(text: str | None) -> list[str]:
    """Split comma-separated IR operands."""
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def value_operands(lines: list[str]) -> Counter[str]:
    """Count textual value uses for simple temp register reuse."""
    uses: Counter[str] = Counter()
    for line in lines:
        line = clean_line(line)
        if not line or line.startswith("kernel ") or line.startswith("label "):
            continue

        match = IR_ASSIGN_RE.match(line)
        if match:
            _, _, rest = match.groups()
            for operand in split_operands(rest):
                if operand.startswith("%"):
                    uses[operand] += 1
            continue

        parts = line.split(None, 1)
        if len(parts) == 2:
            for operand in split_operands(parts[1]):
                if operand.startswith("%"):
                    uses[operand] += 1
    return uses


def clean_line(line: str) -> str:
    """Remove comments and whitespace from one IR line."""
    return line.split("#", 1)[0].strip()


def ir_to_isa(ir_text: str) -> str:
    """Lower a complete Mini-GPU IR module to ISA assembly."""
    lines = [clean_line(line) for line in ir_text.splitlines()]
    lines = [line for line in lines if line]
    allocator = RegisterAllocator(value_operands(lines))
    asm: list[str] = []

    for line in lines:
        if line.startswith("kernel "):
            name = line.split(None, 1)[1]
            if asm:
                asm.append("")
            asm.append(f".kernel {name}")
            continue

        if line.startswith("label "):
            label = line.split(None, 1)[1]
            asm.append(f"{label}:")
            continue

        match = IR_ASSIGN_RE.match(line)
        if match:
            dst, op, rest = match.groups()
            lower_assignment(dst, op, split_operands(rest), allocator, asm)
            continue

        lower_statement(line, allocator, asm)

    return "\n".join(asm)


lower_ir = ir_to_isa


def lower_assignment(
    dst: str,
    op: str,
    operands: list[str],
    allocator: RegisterAllocator,
    asm: list[str],
) -> None:
    """Lower one assigning IR instruction."""
    rd = allocator.define(dst)

    if op == "arg":
        if len(operands) != 1:
            raise IsaLoweringError(f"arg expects one operand: {dst} = {op} {operands}")
        asm.append(f"  LDC {rd}, ARG_{operands[0].upper()}")
        return

    if op == "const":
        if len(operands) != 1:
            raise IsaLoweringError(f"const expects one operand: {dst} = {op} {operands}")
        asm.append(f"  MOVI {rd}, {operands[0]}")
        return

    if op in THREAD_OPS:
        if operands:
            raise IsaLoweringError(f"{op} expects no operands")
        asm.append(f"  {THREAD_OPS[op]} {rd}")
        return

    if op == "load_global":
        if len(operands) != 1:
            raise IsaLoweringError("load_global expects one address operand")
        rs = allocator.use(operands[0])
        asm.append(f"  LDG {rd}, [{rs} + 0]")
        allocator.release_after_use(operands[0])
        return

    if op == "load_shared":
        if len(operands) != 1:
            raise IsaLoweringError("load_shared expects one address operand")
        rs = allocator.use(operands[0])
        asm.append(f"  LDS {rd}, [{rs} + 0]")
        allocator.release_after_use(operands[0])
        return

    if op == "not":
        if len(operands) != 1:
            raise IsaLoweringError("not expects one operand")
        rs = allocator.use(operands[0])
        asm.append(f"  NOT {rd}, {rs}")
        allocator.release_after_use(operands[0])
        return

    if op == "mov":
        if len(operands) != 1:
            raise IsaLoweringError("mov expects one operand")
        rs = allocator.use(operands[0])
        asm.append(f"  MOV {rd}, {rs}")
        allocator.release_after_use(operands[0])
        return

    opcode = VALUE_OPS.get(op) or EXTENSION_VALUE_OPS.get(op)
    if opcode:
        if len(operands) != 2:
            raise IsaLoweringError(f"{op} expects two operands")
        rs1, scratch1 = operand_reg(operands[0], allocator, asm)
        rs2, scratch2 = operand_reg(operands[1], allocator, asm)
        asm.append(f"  {opcode} {rd}, {rs1}, {rs2}")
        release_operand(operands[0], scratch1, allocator)
        release_operand(operands[1], scratch2, allocator)
        return

    raise IsaLoweringError(f"unsupported IR assignment op: {op}")


def operand_reg(
    operand: str,
    allocator: RegisterAllocator,
    asm: list[str],
) -> tuple[str, str | None]:
    """Return a register for a value operand or literal immediate."""
    if operand.startswith("%"):
        return allocator.use(operand), None

    scratch = allocator.acquire_scratch()
    asm.append(f"  MOVI {scratch}, {operand}")
    return scratch, scratch


def release_operand(
    operand: str,
    scratch: str | None,
    allocator: RegisterAllocator,
) -> None:
    """Release a temporary or scratch operand after use."""
    if scratch is not None:
        allocator.release_scratch(scratch)
        return
    allocator.release_after_use(operand)


def lower_statement(line: str, allocator: RegisterAllocator, asm: list[str]) -> None:
    """Lower one non-assigning IR instruction."""
    parts = line.split(None, 1)
    op = parts[0]
    operands = split_operands(parts[1] if len(parts) > 1 else None)

    if op == "store_global":
        if len(operands) != 2:
            raise IsaLoweringError("store_global expects address and value operands")
        addr = allocator.use(operands[0])
        value = allocator.use(operands[1])
        asm.append(f"  STG [{addr} + 0], {value}")
        allocator.release_after_use(operands[0])
        allocator.release_after_use(operands[1])
        return

    if op == "store_shared":
        if len(operands) != 2:
            raise IsaLoweringError("store_shared expects address and value operands")
        addr = allocator.use(operands[0])
        value = allocator.use(operands[1])
        asm.append(f"  STS [{addr} + 0], {value}")
        allocator.release_after_use(operands[0])
        allocator.release_after_use(operands[1])
        return

    if op == "pred_begin":
        if len(operands) != 1:
            raise IsaLoweringError("pred_begin expects one condition operand")
        cond = allocator.use(operands[0])
        asm.append("  PUSHM")
        asm.append(f"  PRED {cond}")
        allocator.release_after_use(operands[0])
        return

    if op == "pred_end":
        if operands:
            raise IsaLoweringError("pred_end expects no operands")
        asm.append("  POPM")
        return

    if op == "branch":
        if len(operands) != 1:
            raise IsaLoweringError("branch expects one label operand")
        asm.append(f"  BRA {operands[0]}")
        return

    if op == "branch_zero":
        if len(operands) != 2:
            raise IsaLoweringError("branch_zero expects condition and label operands")
        cond = allocator.use(operands[0])
        asm.append(f"  BZ {cond}, {operands[1]}")
        allocator.release_after_use(operands[0])
        return

    if op == "branch_nzero":
        if len(operands) != 2:
            raise IsaLoweringError("branch_nzero expects condition and label operands")
        cond = allocator.use(operands[0])
        asm.append(f"  BNZ {cond}, {operands[1]}")
        allocator.release_after_use(operands[0])
        return

    if op == "barrier":
        if operands:
            raise IsaLoweringError("barrier expects no operands")
        asm.append("  BAR")
        return

    if op == "return":
        if operands:
            raise IsaLoweringError("return expects no operands")
        asm.append("  EXIT")
        return

    raise IsaLoweringError(f"unsupported IR statement op: {op}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lower Mini-GPU IR to Mini-GPU ISA assembly.")
    parser.add_argument("ir", type=Path, help="Mini-GPU IR text file")
    parser.add_argument("-o", "--output", type=Path, help="Write ISA assembly to this file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asm = ir_to_isa(args.ir.read_text(encoding="utf-8"))

    if args.output:
        args.output.write_text(asm + "\n", encoding="utf-8")
    else:
        print(asm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
