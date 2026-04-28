#!/usr/bin/env python3
"""Encode Mini-GPU ISA assembly into 32-bit instruction words."""

from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path


OPCODES = {
    "NOP": 0x00,
    "MOV": 0x01,
    "MOVI": 0x02,
    "LDC": 0x03,
    "ADD": 0x04,
    "ADDI": 0x05,
    "SUB": 0x06,
    "SUBI": 0x07,
    "MUL": 0x08,
    "MULI": 0x09,
    "DIV": 0x0A,
    "MOD": 0x0B,
    "AND": 0x0C,
    "ANDI": 0x0D,
    "OR": 0x0E,
    "ORI": 0x0F,
    "XOR": 0x10,
    "XORI": 0x11,
    "NOT": 0x12,
    "SHL": 0x13,
    "SHLI": 0x14,
    "SHR": 0x15,
    "SHRI": 0x16,
    "SLT": 0x17,
    "SLE": 0x18,
    "SGT": 0x19,
    "SGE": 0x1A,
    "SEQ": 0x1B,
    "SNE": 0x1C,
    "LDG": 0x20,
    "STG": 0x21,
    "LDS": 0x22,
    "STS": 0x23,
    "TID": 0x28,
    "TIDX": 0x29,
    "BID": 0x2A,
    "BDIM": 0x2B,
    "GDIM": 0x2C,
    "LID": 0x2D,
    "WID": 0x2E,
    "PUSHM": 0x30,
    "PRED": 0x31,
    "POPM": 0x32,
    "PREDN": 0x33,
    "BRA": 0x38,
    "BZ": 0x39,
    "BNZ": 0x3A,
    "BAR": 0x3B,
    "EXIT": 0x3C,
}

RRR_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
    "AND",
    "OR",
    "XOR",
    "SHL",
    "SHR",
    "SLT",
    "SLE",
    "SGT",
    "SGE",
    "SEQ",
    "SNE",
}

RRI_OPS = {"ADDI", "SUBI", "MULI", "ANDI", "ORI", "XORI", "SHLI", "SHRI"}
THREAD_OPS = {"TID", "TIDX", "BID", "BDIM", "GDIM", "LID", "WID"}
ZERO_OPS = {"NOP", "PUSHM", "POPM", "BAR", "EXIT"}
MEM_RE = re.compile(r"^\[(r\d+)\s*\+\s*([^\]]+)\]$")


class EncodeError(Exception):
    """Raised when assembly cannot be encoded."""

    pass


@dataclass
class AssemblerState:
    labels: dict[str, int] = field(default_factory=dict)
    constants: dict[str, int] = field(default_factory=dict)
    next_arg_id: int = 0
    next_shared_id: int = 32
    next_const_id: int = 64

    def const_id(self, name: str) -> int:
        """Assign stable IDs to ARG_*, SHARED_*, and other LDC symbols."""
        if is_int(name):
            return parse_int(name)
        if name in self.constants:
            return self.constants[name]
        if name.startswith("ARG_"):
            value = self.next_arg_id
            self.next_arg_id += 1
        elif name.startswith("SHARED_"):
            value = self.next_shared_id
            self.next_shared_id += 1
        else:
            value = self.next_const_id
            self.next_const_id += 1
        self.constants[name] = value
        return value


def clean_line(line: str) -> str:
    """Strip comments and surrounding whitespace."""
    return line.split("#", 1)[0].strip()


def parse_program(asm_text: str) -> tuple[list[str], AssemblerState]:
    """Collect labels and return only real instruction lines."""
    state = AssemblerState()
    instructions: list[str] = []

    for raw_line in asm_text.splitlines():
        line = clean_line(raw_line)
        if not line or line.startswith(".kernel"):
            continue
        if line.endswith(":"):
            state.labels[line[:-1]] = len(instructions)
            continue
        instructions.append(line)

    return instructions, state


def isa_to_words(asm_text: str) -> list[int]:
    """Encode assembly text into 32-bit instruction words."""
    instructions, state = parse_program(asm_text)
    return [encode_instruction(line, pc, state) for pc, line in enumerate(instructions)]


def isa_to_hex(asm_text: str) -> str:
    """Encode assembly text as newline-separated 8-digit hex words."""
    return "\n".join(f"{word:08X}" for word in isa_to_words(asm_text))


def isa_to_binary(asm_text: str) -> bytes:
    """Encode assembly text as big-endian raw instruction bytes."""
    return b"".join(struct.pack(">I", word) for word in isa_to_words(asm_text))


def encode_instruction(line: str, pc: int, state: AssemblerState) -> int:
    """Encode one instruction line."""
    op, operands = split_instruction(line)
    if op not in OPCODES:
        raise EncodeError(f"unknown opcode: {op}")

    if op in ZERO_OPS:
        expect_count(op, operands, 0)
        return pack(op)

    if op in THREAD_OPS:
        expect_count(op, operands, 1)
        return pack(op, rd=reg(operands[0]))

    if op == "MOVI":
        expect_count(op, operands, 2)
        return pack(op, rd=reg(operands[0]), imm=parse_imm14(operands[1]))

    if op == "LDC":
        expect_count(op, operands, 2)
        return pack(op, rd=reg(operands[0]), imm=parse_imm14(state.const_id(operands[1])))

    if op in RRR_OPS:
        expect_count(op, operands, 3)
        return pack(op, rd=reg(operands[0]), rs1=reg(operands[1]), rs2=reg(operands[2]))

    if op == "MOV":
        expect_count(op, operands, 2)
        return pack(op, rd=reg(operands[0]), rs1=reg(operands[1]))

    if op in RRI_OPS:
        expect_count(op, operands, 3)
        return pack(op, rd=reg(operands[0]), rs1=reg(operands[1]), imm=parse_imm14(operands[2]))

    if op == "NOT":
        expect_count(op, operands, 2)
        return pack(op, rd=reg(operands[0]), rs1=reg(operands[1]))

    if op in {"LDG", "LDS"}:
        expect_count(op, operands, 2)
        base, imm = parse_memory(operands[1])
        return pack(op, rd=reg(operands[0]), rs1=base, imm=imm)

    if op in {"STG", "STS"}:
        expect_count(op, operands, 2)
        base, imm = parse_memory(operands[0])
        return pack(op, rs1=base, rs2=reg(operands[1]), imm=imm)

    if op in {"PRED", "PREDN"}:
        expect_count(op, operands, 1)
        return pack(op, rs1=reg(operands[0]))

    if op == "BRA":
        expect_count(op, operands, 1)
        return pack(op, imm=branch_offset(operands[0], pc, state))

    if op in {"BZ", "BNZ"}:
        expect_count(op, operands, 2)
        return pack(op, rs1=reg(operands[0]), imm=branch_offset(operands[1], pc, state))

    raise EncodeError(f"unsupported opcode: {op}")


def split_instruction(line: str) -> tuple[str, list[str]]:
    """Split an instruction into opcode and operands."""
    if " " not in line:
        return line.upper(), []
    op, rest = line.split(None, 1)
    return op.upper(), [operand.strip() for operand in rest.split(",")]


def pack(op: str, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int = 0) -> int:
    """Pack fields into the v0 32-bit encoding."""
    for field_name, value in (("rd", rd), ("rs1", rs1), ("rs2", rs2)):
        if not 0 <= value <= 15:
            raise EncodeError(f"{field_name} out of range: {value}")
    return (OPCODES[op] << 26) | (rd << 22) | (rs1 << 18) | (rs2 << 14) | encode_imm14(imm)


def reg(text: str) -> int:
    """Parse r0..r15."""
    text = text.strip().lower()
    if not text.startswith("r") or not text[1:].isdigit():
        raise EncodeError(f"expected register, got {text!r}")
    value = int(text[1:])
    if not 0 <= value <= 15:
        raise EncodeError(f"register out of range: {text}")
    return value


def parse_memory(text: str) -> tuple[int, int]:
    """Parse [rX + imm]."""
    match = MEM_RE.match(text.strip())
    if not match:
        raise EncodeError(f"expected memory operand [rX + imm], got {text!r}")
    return reg(match.group(1)), parse_imm14(match.group(2))


def branch_offset(label: str, pc: int, state: AssemblerState) -> int:
    """Return PC-relative branch offset from the next instruction."""
    if label not in state.labels:
        raise EncodeError(f"unknown label: {label}")
    return parse_imm14(state.labels[label] - (pc + 1))


def parse_imm14(value: int | str) -> int:
    """Parse and range-check a signed 14-bit immediate."""
    parsed = value if isinstance(value, int) else parse_int(value)
    if not -(1 << 13) <= parsed < (1 << 13):
        raise EncodeError(f"immediate out of signed 14-bit range: {parsed}")
    return parsed


def encode_imm14(value: int) -> int:
    """Encode a signed immediate into 14 bits."""
    parse_imm14(value)
    return value & 0x3FFF


def parse_int(text: str) -> int:
    """Parse decimal or 0x-prefixed integers."""
    return int(str(text), 0)


def is_int(text: str) -> bool:
    """Return true when text parses as an integer literal."""
    try:
        parse_int(text)
    except ValueError:
        return False
    return True


def expect_count(op: str, operands: list[str], count: int) -> None:
    """Validate operand count."""
    if len(operands) != count:
        raise EncodeError(f"{op} expects {count} operands, got {len(operands)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode Mini-GPU ISA assembly.")
    parser.add_argument("isa", type=Path, help="Mini-GPU ISA assembly file")
    parser.add_argument("-o", "--output", type=Path, help="Output file")
    parser.add_argument(
        "--format",
        choices=("hex", "binary"),
        default="hex",
        help="Output format (default: hex)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asm_text = args.isa.read_text(encoding="utf-8")

    if args.format == "binary":
        data = isa_to_binary(asm_text)
        if args.output:
            args.output.write_bytes(data)
        else:
            raise SystemExit("error: binary output requires -o")
    else:
        text = isa_to_hex(asm_text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
