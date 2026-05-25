#!/usr/bin/env python3
"""Lower Mini-GPU IR text into Mini-GPU ISA assembly."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


IR_ASSIGN_RE = re.compile(r"^(%[\w.]+)\s*=\s*([\w.]+)(?:\s+(.*))?$")

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

FLOAT_PREFIX_OPS = {
    "add": "FADD",
    "sub": "FSUB",
    "mul": "FMUL",
    "div": "FDIV",
}
SHARED_HELPER_SPILL_SLOT_BASE = 512
SHARED_HELPER_SPILL_SLOT_STRIDE = 64

CONVERT_OPS = {
    "ftoi": "FTOI",
    "itof": "ITOF",
}

COMPARE_OPS = {"lt", "le", "gt", "ge", "eq", "ne"}


class IsaLoweringError(Exception):
    """Raised when IR cannot be represented in the current ISA assembly."""

    pass


@dataclass
class RegisterAllocator:
    remaining_uses: Counter[str]
    value_regs: dict[str, str] = field(default_factory=dict)
    free_regs: list[str] = field(default_factory=lambda: [f"r{i}" for i in range(13)])
    spilled_values: dict[str, int] = field(default_factory=dict)
    value_depths: dict[str, int] = field(default_factory=dict)
    persistent_values: set[str] = field(default_factory=set)
    memory_backed_values: set[str] = field(default_factory=set)
    next_spill_slot: int = 0
    spill_base_reg: str = "r15"
    spill_addr_reg: str = "r14"
    scratch_regs: list[str] = field(default_factory=lambda: ["r13"])
    pinned_values: set[str] = field(default_factory=set)
    pred_depth: int = 0

    def define(self, value: str, asm: list[str]) -> str:
        """Assign a physical register to a newly defined IR value."""
        if value.startswith("%slot"):
            self.persistent_values.add(value)
            self.memory_backed_values.add(value)

        old_reg = self.value_regs.pop(value, None)
        if old_reg is not None:
            self.value_regs[value] = old_reg
            self.value_depths[value] = self.pred_depth
            return old_reg

        if value not in self.memory_backed_values:
            self.spilled_values.pop(value, None)
        reg = self.acquire_value_reg(asm)
        self.value_regs[value] = reg
        self.value_depths[value] = self.pred_depth
        return reg

    def use(self, value: str, asm: list[str]) -> str:
        """Return the register containing an existing IR value, reloading if spilled."""
        if value in self.value_regs:
            return self.value_regs[value]
        if value in self.spilled_values:
            reg = self.acquire_value_reg(asm)
            self.emit_spill_load(reg, value, asm)
            self.value_regs[value] = reg
            self.value_depths[value] = self.pred_depth
            return reg
        raise IsaLoweringError(f"use of undefined IR value: {value}")

    def acquire_value_reg(self, asm: list[str], exclude: set[str] | None = None) -> str:
        """Get a value register, spilling an existing value if needed."""
        exclude = (exclude or set()) | self.pinned_values
        if not self.free_regs:
            self.spill_one(asm, exclude)
        if not self.free_regs:
            raise IsaLoweringError("ran out of registers; no spill candidate was available")
        return self.free_regs.pop(0)

    def spill_one(self, asm: list[str], exclude: set[str] | None = None) -> None:
        """Move one resident value to the compiler spill area."""
        exclude = (exclude or set()) | self.pinned_values
        candidate: str | None = None
        for value in self.value_regs:
            if value not in exclude and self.value_depths.get(value, 0) == self.pred_depth:
                candidate = value
                break
        if candidate is None:
            return
        reg = self.value_regs.pop(candidate)
        self.emit_spill_store(candidate, reg, asm)
        self.free_regs.insert(0, reg)

    def spill_all(self, asm: list[str], exclude: set[str] | None = None) -> None:
        """Spill all currently resident values before a helper call."""
        exclude = (exclude or set()) | self.pinned_values
        for value, reg in list(self.value_regs.items()):
            if value in exclude:
                continue
            self.value_regs.pop(value, None)
            self.emit_spill_store(value, reg, asm)
            self.value_depths.pop(value, None)
            self.free_regs.insert(0, reg)

    def spill_slot(self, value: str) -> int:
        """Return the spill slot assigned to an IR value."""
        slot = self.spilled_values.get(value)
        if slot is not None:
            return slot
        slot = self.next_spill_slot
        self.next_spill_slot += 1
        self.spilled_values[value] = slot
        return slot

    def emit_spill_base(self, asm: list[str]) -> None:
        """Materialize the compiler-reserved shared spill base."""
        asm.append(f"  LDC {self.spill_base_reg}, SHARED_SPILL_BASE")

    def emit_spill_addr(self, slot: int, asm: list[str]) -> None:
        """Materialize a per-lane spill address for one logical spill slot."""
        self.emit_spill_base(asm)
        asm.append(f"  TIDX {self.spill_addr_reg}")
        if slot:
            asm.append(f"  ADDI {self.spill_addr_reg}, {self.spill_addr_reg}, {slot * 4}")
        asm.append(f"  ADD {self.spill_addr_reg}, {self.spill_base_reg}, {self.spill_addr_reg}")

    def emit_spill_store(self, value: str, reg: str, asm: list[str]) -> None:
        """Store one live register value into its spill slot."""
        slot = self.spill_slot(value)
        self.emit_spill_addr(slot, asm)
        asm.append(f"  STS [{self.spill_addr_reg} + 0], {reg}")

    def emit_spill_load(self, reg: str, value: str, asm: list[str]) -> None:
        """Reload one spilled value from its spill slot."""
        slot = self.spill_slot(value)
        self.emit_spill_addr(slot, asm)
        asm.append(f"  LDS {reg}, [{self.spill_addr_reg} + 0]")

    def release_after_use(self, value: str) -> None:
        """Free value registers or spill slots after their final textual use."""
        if not value.startswith("%"):
            return
        if value in self.memory_backed_values:
            reg = self.value_regs.pop(value, None)
            if reg is not None:
                self.free_regs.insert(0, reg)
            self.value_depths.pop(value, None)
            return
        if value in self.persistent_values:
            return

        self.remaining_uses[value] -= 1
        if self.remaining_uses[value] > 0:
            return

        reg = self.value_regs.pop(value, None)
        if reg is not None:
            self.free_regs.insert(0, reg)
        self.spilled_values.pop(value, None)
        self.value_depths.pop(value, None)

    def acquire_scratch(self) -> str:
        """Reserve a short-lived register for materialized immediates."""
        if not self.scratch_regs:
            raise IsaLoweringError("ran out of compiler scratch registers")
        return self.scratch_regs.pop(0)

    def release_scratch(self, reg: str) -> None:
        """Return a scratch register to the free list."""
        self.scratch_regs.insert(0, reg)

    def pin(self, value: str) -> None:
        """Prevent a value from being chosen as a spill victim."""
        self.pinned_values.add(value)

    def unpin(self, value: str) -> None:
        """Allow a value to be spilled again."""
        self.pinned_values.discard(value)

    def enter_predicate(self) -> None:
        """Track that following instructions execute under a narrower lane mask."""
        self.pred_depth += 1

    def exit_predicate(self) -> None:
        """Track that following instructions execute under the restored lane mask."""
        if self.pred_depth > 0:
            self.pred_depth -= 1
        for value, depth in list(self.value_depths.items()):
            if depth > self.pred_depth:
                self.value_depths[value] = self.pred_depth


def split_operands(text: str | None) -> list[str]:
    """Split comma-separated IR operands."""
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def is_temp_value(value: str) -> bool:
    """Return true for compiler-generated temporaries like %t12."""
    return len(value) > 2 and value[0:2] == "%t" and value[2:].isdigit()


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


def split_typed_op(op: str) -> tuple[str, str | None]:
    """Split IR ops like add.i16 or load_global.fp32."""
    if "." not in op:
        return op, None
    base, fmt = op.split(".", 1)
    return base, fmt


def isa_format(fmt: str | None, default: str = "i32") -> str:
    """Map IR format names to ISA suffix names."""
    return (fmt or default).upper()


def isa_value_opcode(base_op: str, fmt: str | None) -> str | None:
    """Return an ISA opcode for a typed or untyped value op."""
    effective_fmt = fmt or "i32"
    if base_op in COMPARE_OPS:
        opcode = VALUE_OPS.get(base_op)
        return f"{opcode}.{isa_format(fmt)}" if fmt is not None else opcode

    if effective_fmt.startswith("fp"):
        opcode = FLOAT_PREFIX_OPS.get(base_op)
        return f"{opcode}.{isa_format(effective_fmt)}" if opcode else None

    opcode = VALUE_OPS.get(base_op) or EXTENSION_VALUE_OPS.get(base_op)
    if opcode is None:
        return None
    if fmt is not None:
        return f"{opcode}.{isa_format(fmt)}"
    return opcode


def ir_to_isa(ir_text: str) -> str:
    """Lower a complete Mini-GPU IR module to ISA assembly."""
    lines = [clean_line(line) for line in ir_text.splitlines()]
    lines = [line for line in lines if line]
    asm: list[str] = []

    kernel_lines: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("kernel "):
            if current:
                kernel_lines.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        kernel_lines.append(current)

    helper_spill_bases: dict[str, int] = {}
    for section in kernel_lines:
        allocator = RegisterAllocator(value_operands(section))
        section_name = section[0].split(None, 1)[1] if section and section[0].startswith("kernel ") else ""
        if section_uses_call_abi(section):
            helper_spill_bases.setdefault(
                section_name,
                SHARED_HELPER_SPILL_SLOT_BASE
                + len(helper_spill_bases) * SHARED_HELPER_SPILL_SLOT_STRIDE,
            )
            allocator.next_spill_slot = helper_spill_bases[section_name]

        for line in section:
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


def section_uses_call_abi(section: list[str]) -> bool:
    """Return true for lowered shared helper sections using call ABI pseudo-ops."""
    return any("call_arg" in line or line.startswith("call_return") for line in section)


def lower_assignment(
    dst: str,
    op: str,
    operands: list[str],
    allocator: RegisterAllocator,
    asm: list[str],
) -> None:
    """Lower one assigning IR instruction."""
    base_op, fmt = split_typed_op(op)
    if base_op == "call":
        lower_call_assignment(dst, fmt, operands, allocator, asm)
        return

    rd = allocator.define(dst, asm)
    allocator.pin(dst)

    try:
        if op == "arg":
            if len(operands) != 1:
                raise IsaLoweringError(f"arg expects one operand: {dst} = {op} {operands}")
            allocator.persistent_values.add(dst)
            allocator.memory_backed_values.add(dst)
            asm.append(f"  LDC {rd}, ARG_{operands[0].upper()}")
            allocator.emit_spill_store(dst, rd, asm)
            allocator.value_regs.pop(dst, None)
            allocator.value_depths.pop(dst, None)
            allocator.free_regs.insert(0, rd)
            return

        if op == "shared":
            if len(operands) != 1:
                raise IsaLoweringError(f"shared expects one operand: {dst} = {op} {operands}")
            asm.append(f"  LDC {rd}, SHARED_{operands[0].upper()}")
            return

        if op == "const":
            if len(operands) != 1:
                raise IsaLoweringError(f"const expects one operand: {dst} = {op} {operands}")
            emit_load_immediate(rd, operands[0], asm)
            return

        if op in THREAD_OPS:
            if operands:
                raise IsaLoweringError(f"{op} expects no operands")
            asm.append(f"  {THREAD_OPS[op]} {rd}")
            return

        if base_op == "arg":
            if len(operands) != 1:
                raise IsaLoweringError(f"arg expects one operand: {dst} = {op} {operands}")
            allocator.persistent_values.add(dst)
            allocator.memory_backed_values.add(dst)
            asm.append(f"  LDC {rd}, ARG_{operands[0].upper()}")
            allocator.emit_spill_store(dst, rd, asm)
            allocator.value_regs.pop(dst, None)
            allocator.value_depths.pop(dst, None)
            allocator.free_regs.insert(0, rd)
            return

        if base_op == "call_arg":
            if len(operands) != 1:
                raise IsaLoweringError("call_arg expects one argument index")
            index = parse_int_literal(operands[0])
            if index < 0 or index > 3:
                raise IsaLoweringError("call_arg supports argument registers r0..r3")
            allocator.persistent_values.add(dst)
            allocator.memory_backed_values.add(dst)
            asm.append(f"  MOV {rd}, r{index}")
            allocator.emit_spill_store(dst, rd, asm)
            allocator.value_regs.pop(dst, None)
            allocator.value_depths.pop(dst, None)
            allocator.free_regs.insert(0, rd)
            return

        if base_op == "shared":
            if len(operands) != 1:
                raise IsaLoweringError(f"shared expects one operand: {dst} = {op} {operands}")
            asm.append(f"  LDC {rd}, SHARED_{operands[0].upper()}")
            return

        if base_op == "const":
            if len(operands) != 1:
                raise IsaLoweringError(f"const expects one operand: {dst} = {op} {operands}")
            emit_load_immediate(rd, operands[0], asm)
            return

        if base_op in THREAD_OPS:
            if operands:
                raise IsaLoweringError(f"{base_op} expects no operands")
            asm.append(f"  {THREAD_OPS[base_op]} {rd}")
            return

        if base_op == "load_global":
            if len(operands) != 1:
                raise IsaLoweringError("load_global expects one address operand")
            rs, scratch = operand_reg(operands[0], allocator, asm)
            suffix = f".{isa_format(fmt)}" if fmt is not None else ""
            asm.append(f"  LDG{suffix} {rd}, [{rs} + 0]")
            release_operand(operands[0], scratch, allocator)
            return

        if base_op == "load_shared":
            if len(operands) != 1:
                raise IsaLoweringError("load_shared expects one address operand")
            rs, scratch = operand_reg(operands[0], allocator, asm)
            suffix = f".{isa_format(fmt)}" if fmt is not None else ""
            asm.append(f"  LDS{suffix} {rd}, [{rs} + 0]")
            release_operand(operands[0], scratch, allocator)
            return

        if base_op == "not":
            if len(operands) != 1:
                raise IsaLoweringError("not expects one operand")
            rs, scratch = operand_reg(operands[0], allocator, asm)
            asm.append(f"  NOT {rd}, {rs}")
            release_operand(operands[0], scratch, allocator)
            return

        if base_op == "mov":
            if len(operands) != 1:
                raise IsaLoweringError("mov expects one operand")
            rs, scratch = operand_reg(operands[0], allocator, asm)
            asm.append(f"  MOV {rd}, {rs}")
            release_operand(operands[0], scratch, allocator)
            return

        if base_op in CONVERT_OPS:
            if len(operands) != 1:
                raise IsaLoweringError(f"{base_op} expects one operand")
            rs, scratch = operand_reg(operands[0], allocator, asm)
            suffix = f".{isa_format(fmt, 'fp32' if base_op == 'ftoi' else 'i32')}"
            asm.append(f"  {CONVERT_OPS[base_op]}{suffix} {rd}, {rs}")
            release_operand(operands[0], scratch, allocator)
            return

        opcode = isa_value_opcode(base_op, fmt)
        if opcode:
            if len(operands) != 2:
                raise IsaLoweringError(f"{op} expects two operands")
            rs1, scratch1 = operand_reg(operands[0], allocator, asm)
            if operands[0].startswith("%") and scratch1 is None:
                allocator.pin(operands[0])
            rs2, scratch2 = operand_reg(operands[1], allocator, asm)
            asm.append(f"  {opcode} {rd}, {rs1}, {rs2}")
            if operands[0].startswith("%") and scratch1 is None:
                allocator.unpin(operands[0])
            release_operand(operands[0], scratch1, allocator)
            release_operand(operands[1], scratch2, allocator)
            return

        raise IsaLoweringError(f"unsupported IR assignment op: {op}")
    finally:
        if dst in allocator.memory_backed_values and dst in allocator.value_regs:
            dst_reg = allocator.value_regs.pop(dst)
            allocator.emit_spill_store(dst, dst_reg, asm)
            allocator.value_depths.pop(dst, None)
            allocator.free_regs.insert(0, dst_reg)
        allocator.unpin(dst)


def lower_call_assignment(
    dst: str,
    fmt: str | None,
    operands: list[str],
    allocator: RegisterAllocator,
    asm: list[str],
) -> None:
    """Lower a shared helper call using r0..rN args and r0 as return register."""
    if len(operands) < 1:
        raise IsaLoweringError("call expects helper name")
    helper, *args = operands
    if len(args) > 4:
        raise IsaLoweringError("shared helper calls support up to four arguments")
    allocator.spill_all(asm)
    abi_regs = [f"r{idx}" for idx in range(len(args))]
    saved_free_regs = list(allocator.free_regs)
    allocator.free_regs = [reg for reg in allocator.free_regs if reg not in abi_regs]
    try:
        for index, arg in enumerate(args):
            rs, scratch = operand_reg(arg, allocator, asm)
            if rs != abi_regs[index]:
                asm.append(f"  MOV {abi_regs[index]}, {rs}")
            release_operand(arg, scratch, allocator)
    finally:
        allocator.free_regs = saved_free_regs
    asm.append(f"  CALL @{helper}")
    rd = allocator.define(dst, asm)
    asm.append(f"  MOV {rd}, r0")


def operand_reg(
    operand: str,
    allocator: RegisterAllocator,
    asm: list[str],
) -> tuple[str, str | None]:
    """Return a register for a value operand or literal immediate."""
    if operand.startswith("%"):
        return allocator.use(operand, asm), None

    scratch = allocator.acquire_scratch()
    emit_load_immediate(scratch, operand, asm)
    return scratch, scratch


def parse_int_literal(value: str) -> int:
    """Parse decimal/hex integer literal spellings from IR."""
    text = value.strip()
    text = re.sub(r"[uUlL]+$", "", text)
    return int(text, 0)


def fits_imm14(value: int) -> bool:
    """Return true when a value fits the signed MOVI immediate."""
    return -(1 << 13) <= value < (1 << 13)


def emit_load_immediate(reg: str, value: str, asm: list[str]) -> None:
    """Load a small or full 32-bit integer literal into a register."""
    parsed = parse_int_literal(value)
    if fits_imm14(parsed):
        asm.append(f"  MOVI {reg}, {parsed}")
        return

    word = parsed & 0xffffffff
    bytes_be = [
        (word >> 24) & 0xff,
        (word >> 16) & 0xff,
        (word >> 8) & 0xff,
        word & 0xff,
    ]
    asm.append(f"  MOVI {reg}, {bytes_be[0]}")
    for byte in bytes_be[1:]:
        asm.append(f"  SHLI {reg}, {reg}, 8")
        if byte:
            asm.append(f"  ORI {reg}, {reg}, {byte}")


def release_operand(
    operand: str,
    scratch: str | None,
    allocator: RegisterAllocator,
) -> None:
    """Release a temporary or scratch operand after use."""
    if scratch is not None:
        allocator.release_scratch(scratch)
        allocator.release_after_use(operand)
        return
    allocator.release_after_use(operand)


def lower_statement(line: str, allocator: RegisterAllocator, asm: list[str]) -> None:
    """Lower one non-assigning IR instruction."""
    parts = line.split(None, 1)
    op = parts[0]
    base_op, fmt = split_typed_op(op)
    operands = split_operands(parts[1] if len(parts) > 1 else None)

    if base_op == "store_global":
        if len(operands) != 2:
            raise IsaLoweringError("store_global expects address and value operands")
        addr, scratch1 = operand_reg(operands[0], allocator, asm)
        if operands[0].startswith("%") and scratch1 is None:
            allocator.pin(operands[0])
        value, scratch2 = operand_reg(operands[1], allocator, asm)
        suffix = f".{isa_format(fmt)}" if fmt is not None else ""
        asm.append(f"  STG{suffix} [{addr} + 0], {value}")
        if operands[0].startswith("%") and scratch1 is None:
            allocator.unpin(operands[0])
        release_operand(operands[0], scratch1, allocator)
        release_operand(operands[1], scratch2, allocator)
        return

    if base_op == "store_shared":
        if len(operands) != 2:
            raise IsaLoweringError("store_shared expects address and value operands")
        addr, scratch1 = operand_reg(operands[0], allocator, asm)
        if operands[0].startswith("%") and scratch1 is None:
            allocator.pin(operands[0])
        value, scratch2 = operand_reg(operands[1], allocator, asm)
        suffix = f".{isa_format(fmt)}" if fmt is not None else ""
        asm.append(f"  STS{suffix} [{addr} + 0], {value}")
        if operands[0].startswith("%") and scratch1 is None:
            allocator.unpin(operands[0])
        release_operand(operands[0], scratch1, allocator)
        release_operand(operands[1], scratch2, allocator)
        return

    if op == "pred_begin":
        if len(operands) != 1:
            raise IsaLoweringError("pred_begin expects one condition operand")
        cond, scratch = operand_reg(operands[0], allocator, asm)
        asm.append("  PUSHM")
        asm.append(f"  PRED {cond}")
        release_operand(operands[0], scratch, allocator)
        allocator.enter_predicate()
        return

    if op == "pred_end":
        if operands:
            raise IsaLoweringError("pred_end expects no operands")
        allocator.exit_predicate()
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
        cond, scratch = operand_reg(operands[0], allocator, asm)
        asm.append(f"  BZ {cond}, {operands[1]}")
        release_operand(operands[0], scratch, allocator)
        return

    if op == "branch_nzero":
        if len(operands) != 2:
            raise IsaLoweringError("branch_nzero expects condition and label operands")
        cond, scratch = operand_reg(operands[0], allocator, asm)
        asm.append(f"  BNZ {cond}, {operands[1]}")
        release_operand(operands[0], scratch, allocator)
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

    if base_op == "call_return":
        if len(operands) != 1:
            raise IsaLoweringError("call_return expects one value operand")
        value_reg, scratch = operand_reg(operands[0], allocator, asm)
        if value_reg != "r0":
            asm.append(f"  MOV r0, {value_reg}")
        release_operand(operands[0], scratch, allocator)
        asm.append("  RET")
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
