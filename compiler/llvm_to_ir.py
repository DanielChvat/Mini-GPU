#!/usr/bin/env python3
"""Lower Clang-emitted LLVM IR into Mini-GPU IR.

This is intentionally a legalization pass for the current Mini-GPU hardware,
not a general LLVM backend. Clang owns CUDA parsing and C expression semantics;
this pass recognizes the structured IR Clang emits for the supported kernel
subset and preserves Mini-GPU's lane predication model.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ast_to_ir import (
    format_from_qual_type,
    is_packed_format,
    merge_value_formats,
    packed_byte_shift,
    typed_ir_op,
)


DEFINE_RE = re.compile(r"^define\b.*?@([^(]+)\((.*)\)")
LABEL_RE = re.compile(r"^([A-Za-z$._-][\w$._-]*|\d+):")
ASSIGN_RE = re.compile(r"^(%[-\w.]+)\s*=\s*(.+)$")
ALLOCA_RE = re.compile(r"alloca\s+([^,]+)")
LOAD_RE = re.compile(r"load\s+([^,]+),\s+ptr\s+(.+?)(?:,\s|$)")
STORE_RE = re.compile(r"store\s+(.+?)\s+(.+?),\s+ptr\s+(.+?)(?:,\s|$)")
GEP_RE = re.compile(r"getelementptr(?:\s+inbounds)?\s+([^,]+),\s+ptr\s+(.+?),\s+(.+)$")
BR_COND_RE = re.compile(r"br\s+i1\s+([^,]+),\s+label\s+%?([^,\s]+),\s+label\s+%?([^,\s]+)")
BR_RE = re.compile(r"br\s+label\s+%?([^,\s]+)")
CALL_RE = re.compile(r"call\b.*?@([^(]+)\((.*)\)")


INT_OPS = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "sdiv": "div",
    "udiv": "div",
    "srem": "mod",
    "urem": "mod",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shl": "shl",
    "ashr": "shr",
    "lshr": "shr",
}

FLOAT_OPS = {"fadd": "add", "fsub": "sub", "fmul": "mul", "fdiv": "div"}
SHARED_ANNOTATION = "minigpu_shared"

ICMP_OPS = {
    "slt": "lt",
    "ult": "lt",
    "sle": "le",
    "ule": "le",
    "sgt": "gt",
    "ugt": "gt",
    "sge": "ge",
    "uge": "ge",
    "eq": "eq",
    "ne": "ne",
}

FCMP_OPS = {
    "olt": "lt",
    "ult": "lt",
    "ole": "le",
    "ule": "le",
    "ogt": "gt",
    "ugt": "gt",
    "oge": "ge",
    "uge": "ge",
    "oeq": "eq",
    "ueq": "eq",
    "one": "ne",
    "une": "ne",
}


class LlvmLoweringError(Exception):
    """Raised when LLVM IR cannot be represented by Mini-GPU IR."""


@dataclass
class Function:
    symbol: str
    return_format: str
    params: list[str]
    param_formats: list[str]
    blocks: dict[str, list[str]]
    order: list[str]


@dataclass
class KernelMeta:
    name: str
    arg_names: list[str]
    arg_formats: list[str]
    arg_is_pointer: list[bool]


@dataclass
class LowerState:
    function: Function
    meta: KernelMeta
    lines: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    value_types: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)
    slot_types: dict[str, str] = field(default_factory=dict)
    pointers: dict[str, tuple[str, str]] = field(default_factory=dict)
    temp_index: int = 0
    visited: set[str] = field(default_factory=set)
    is_shared_helper: bool = False
    shared_helpers: set[str] = field(default_factory=set)
    shared_helper_returns: dict[str, str] = field(default_factory=dict)

    def emit(self, text: str = "") -> None:
        self.lines.append(text)

    def temp(self) -> str:
        self.temp_index += 1
        return f"%t{self.temp_index}"


def llvm_to_ir(
    llvm_text: str,
    ast_document: dict[str, Any],
    only_kernels: set[str] | None = None,
) -> str:
    """Lower an LLVM IR module into Mini-GPU IR text."""
    functions = parse_functions(llvm_text)
    metas = kernel_metadata(ast_document)
    if only_kernels:
        metas = {
            symbol: meta
            for symbol, meta in metas.items()
            if symbol in only_kernels or meta.name in only_kernels
        }
        if not metas:
            requested = ", ".join(sorted(only_kernels))
            raise LlvmLoweringError(f"no CUDA kernel matched --only-kernel {requested}")
    chunks: list[str] = []
    shared_helpers = shared_helpers_from_annotations(llvm_text)
    helper_returns = {
        name: functions[name].return_format
        for name in shared_helpers
        if name in functions
    }
    reachable_helpers = reachable_shared_helpers(functions, metas.keys(), shared_helpers)
    for symbol, meta in metas.items():
        function = functions.get(symbol)
        if function is None:
            raise LlvmLoweringError(f"missing LLVM function for CUDA kernel {meta.name} ({symbol})")
        chunks.append(lower_function(function, meta, shared_helpers, helper_returns))
    for helper in sorted(reachable_helpers):
        function = functions.get(helper) if helper in reachable_helpers else None
        if function is not None:
            chunks.append(lower_shared_helper(function, helper, shared_helpers, helper_returns))
    return "\n\n".join(chunks)


def reachable_shared_helpers(
    functions: dict[str, Function],
    roots: Any,
    shared_helpers: set[str],
) -> set[str]:
    """Return shared helpers reachable from kernels through direct LLVM calls."""
    reachable: set[str] = set()
    stack = list(roots)
    seen: set[str] = set()
    while stack:
        symbol = stack.pop()
        if symbol in seen:
            continue
        seen.add(symbol)
        function = functions.get(symbol)
        if function is None:
            continue
        for lines in function.blocks.values():
            for line in lines:
                call = CALL_RE.search(line)
                if not call:
                    continue
                callee = call.group(1)
                if callee in shared_helpers:
                    reachable.add(callee)
                    stack.append(callee)
    return reachable


def shared_helpers_from_annotations(text: str) -> set[str]:
    """Return function symbols tagged with MINIGPU_SHARED in source."""
    annotation_globals: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^@([^=\s]+)\s*=.*c\"" + re.escape(SHARED_ANNOTATION) + r"\\00\"", stripped)
        if match:
            annotation_globals.add(match.group(1))

    if not annotation_globals:
        return set()

    helpers: set[str] = set()
    for line in text.splitlines():
        if "@llvm.global.annotations" not in line:
            continue
        for function_name, annotation_name in re.findall(
            r"\{\s*ptr\s+@([^,\s]+),\s*ptr\s+@([^,\s]+),", line
        ):
            if annotation_name in annotation_globals:
                helpers.add(function_name)
    return helpers


def parse_functions(text: str) -> dict[str, Function]:
    functions: dict[str, Function] = {}
    lines = [clean_metadata(line) for line in text.splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        match = DEFINE_RE.match(line)
        if not match:
            index += 1
            continue
        symbol = match.group(1)
        return_format = define_return_format(line, symbol)
        params, param_formats = parse_params(match.group(2))
        blocks: dict[str, list[str]] = {"entry": []}
        order = ["entry"]
        current = "entry"
        index += 1
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped == "}":
                break
            label = LABEL_RE.match(stripped)
            if label:
                current = label.group(1)
                blocks[current] = []
                order.append(current)
            elif stripped and not stripped.startswith(";"):
                blocks[current].append(stripped)
            index += 1
        functions[symbol] = Function(
            symbol=symbol,
            return_format=return_format,
            params=params,
            param_formats=param_formats,
            blocks=blocks,
            order=order,
        )
        index += 1
    return functions


def clean_metadata(line: str) -> str:
    return line.split("!", 1)[0].rstrip()


def define_return_format(line: str, symbol: str) -> str:
    marker = f"@{symbol}"
    prefix = line.split(marker, 1)[0].strip()
    if not prefix:
        return "void"
    return llvm_format(prefix.split()[-1])


def parse_params(text: str) -> tuple[list[str], list[str]]:
    params: list[str] = []
    formats: list[str] = []
    for part in split_operands(text):
        match = re.search(r"(%[-\w.]+)(?:\s+)?$", part.strip())
        if match:
            params.append(match.group(1))
            formats.append(llvm_format(part[:match.start()].strip()))
    return params, formats


def kernel_metadata(document: dict[str, Any]) -> dict[str, KernelMeta]:
    metas: dict[str, KernelMeta] = {}
    for kernel in document.get("kernels", []):
        symbol = kernel.get("mangledName")
        name = kernel.get("minigpuKernelName") or kernel.get("name")
        if not symbol or not name:
            continue
        arg_names: list[str] = []
        arg_formats: list[str] = []
        arg_is_pointer: list[bool] = []
        for child in kernel.get("inner", []):
            if child.get("kind") != "ParmVarDecl":
                continue
            qual_type = type_text(child)
            arg_names.append(child.get("name") or f"arg{len(arg_names)}")
            arg_formats.append(format_from_qual_type(qual_type) or "i32")
            arg_is_pointer.append("*" in qual_type)
        metas[symbol] = KernelMeta(
            name=name,
            arg_names=arg_names,
            arg_formats=arg_formats,
            arg_is_pointer=arg_is_pointer,
        )
    return metas


def lower_function(
    function: Function,
    meta: KernelMeta,
    shared_helpers: set[str],
    shared_helper_returns: dict[str, str],
) -> str:
    state = LowerState(
        function=function,
        meta=meta,
        shared_helpers=shared_helpers,
        shared_helper_returns=shared_helper_returns,
    )
    state.emit(f"kernel {meta.name}")
    for index, llvm_name in enumerate(function.params):
        arg_name = meta.arg_names[index] if index < len(meta.arg_names) else f"arg{index}"
        fmt = meta.arg_formats[index] if index < len(meta.arg_formats) else "i32"
        value = f"%{arg_name}"
        state.values[llvm_name] = value
        state.value_types[value] = "ptr" if is_param_pointer(index, meta) else fmt
        state.emit(f"{value} = arg {arg_name}")
    if function.params:
        state.emit()

    lower_entry(state)
    if state.lines[-1] != "return":
        state.emit("return")
    return "\n".join(state.lines)


def lower_shared_helper(
    function: Function,
    name: str,
    shared_helpers: set[str],
    shared_helper_returns: dict[str, str],
) -> str:
    """Lower a noinline device helper using r0..rN args and r0 return value."""
    arg_names = [f"arg{idx}" for idx in range(len(function.params))]
    arg_formats = function.param_formats or ["fp32" for _ in function.params]
    meta = KernelMeta(
        name=name,
        arg_names=arg_names,
        arg_formats=arg_formats,
        arg_is_pointer=[False for _ in function.params],
    )
    state = LowerState(
        function=function,
        meta=meta,
        is_shared_helper=True,
        shared_helpers=shared_helpers,
        shared_helper_returns=shared_helper_returns,
    )
    state.emit(f"kernel {name}")
    for index, llvm_name in enumerate(function.params):
        fmt = arg_formats[index] if index < len(arg_formats) else "fp32"
        value_name = f"%arg{index}"
        state.values[llvm_name] = value_name
        state.value_types[value_name] = fmt
        state.emit(f"{value_name} = call_arg.{fmt} {index}")
    if function.params:
        state.emit()

    lower_entry(state)
    return "\n".join(state.lines)


def is_param_pointer(index: int, meta: KernelMeta) -> bool:
    return index < len(meta.arg_is_pointer) and meta.arg_is_pointer[index]


def lower_entry(state: LowerState) -> None:
    if state.is_shared_helper:
        emit_block("entry", state, stop=set())
        return

    entry = state.function.blocks["entry"]
    target: str | None = None
    for line in entry:
        term = terminator(line)
        if term:
            if term[0] != "br":
                raise LlvmLoweringError("entry block must branch into kernel body")
            target = term[1]
            break
        lower_instruction(line, state)
    if target is not None:
        emit_block(target, state, stop=set())


def emit_return(term: tuple[str, ...] | None, state: LowerState) -> None:
    """Emit kernel EXIT or helper RET for an LLVM return terminator."""
    if state.is_shared_helper:
        if term is None or len(term) != 2:
            raise LlvmLoweringError(f"shared helper {state.meta.name} must return a value")
        fmt = state.function.return_format
        if fmt == "void":
            fmt = token_type(term[1], state)
        state.emit(f"call_return.{fmt} {value(term[1], state)}")
    else:
        state.emit("return")


def emit_block(label: str, state: LowerState, stop: set[str]) -> None:
    if label in stop:
        return
    if label in state.visited:
        return
    state.visited.add(label)
    state.emit(f"label llvm_{label}")

    instructions, term = split_block(state.function.blocks[label])
    for line in instructions:
        lower_instruction(line, state)

    if term is None or term[0] == "ret":
        emit_return(term, state)
        return
    if term[0] == "br":
        emit_block(term[1], state, stop)
        return

    cond, true_label, false_label = term[1], term[2], term[3]
    if has_backedge_to(label, state.function.blocks) and reaches_avoiding(
        true_label, label, false_label, state.function.blocks, set()
    ):
        emit_loop_tail(label, instructions, cond, true_label, false_label, state)
        emit_block(false_label, state, stop)
        return

    merge = common_merge(true_label, false_label, state.function.blocks)
    if merge is None:
        raise LlvmLoweringError(f"unsupported branch shape in {state.meta.name} at block {label}")

    cond_value = value(cond, state)
    state.emit(f"pred_begin {cond_value}")
    emit_region_until(true_label, merge, label, state)
    state.emit("pred_end")

    if false_label != merge:
        not_cond = state.temp()
        state.value_types[not_cond] = "i32"
        state.emit(f"{not_cond} = eq {cond_value}, 0")
        state.emit(f"pred_begin {not_cond}")
        emit_region_until(false_label, merge, label, state)
        state.emit("pred_end")

    emit_block(merge, state, stop)


def emit_loop_tail(
    header: str,
    header_instructions: list[str],
    cond: str,
    body_label: str,
    exit_label: str,
    state: LowerState,
) -> None:
    cond_value = value(cond, state)
    state.emit(f"pred_begin {cond_value}")
    emit_region_until(body_label, header, header, state)
    state.emit("pred_end")
    for line in header_instructions:
        lower_instruction(line, state)
    next_cond = value(cond, state)
    state.emit(f"branch_nzero {next_cond}, llvm_{header}")


def emit_region_until(
    label: str,
    stop_label: str,
    loop_header: str,
    state: LowerState,
    seen: set[str] | None = None,
) -> None:
    if label == stop_label:
        return
    seen = set() if seen is None else set(seen)
    if label in seen:
        return
    seen.add(label)
    instructions, term = split_block(state.function.blocks[label])
    if term is not None and term[0] == "br_cond" and has_backedge_to(
        label, state.function.blocks
    ) and (
        reaches_avoiding(term[2], label, term[3], state.function.blocks, set())
        or reaches_avoiding(term[3], label, term[2], state.function.blocks, set())
    ) and (not state.lines or state.lines[-1] != f"label llvm_{label}"):
        state.emit(f"label llvm_{label}")
    for line in instructions:
        lower_instruction(line, state)
    if term is None or term[0] == "ret":
        emit_return(term, state)
        return
    if term[0] == "br":
        emit_region_until(term[1], stop_label, loop_header, state, seen)
        return

    cond, true_label, false_label = term[1], term[2], term[3]
    if has_backedge_to(label, state.function.blocks) and reaches_avoiding(
        true_label, label, false_label, state.function.blocks, set()
    ):
        emit_loop_tail(label, instructions, cond, true_label, false_label, state)
        emit_region_until(false_label, stop_label, loop_header, state, seen)
        return

    merge = common_merge(true_label, false_label, state.function.blocks)
    if merge is None:
        raise LlvmLoweringError(f"unsupported nested branch shape at block {label}")
    cond_value = value(cond, state)
    state.emit(f"pred_begin {cond_value}")
    emit_region_until(true_label, merge, loop_header, state, seen)
    state.emit("pred_end")
    if false_label != merge:
        not_cond = state.temp()
        state.value_types[not_cond] = "i32"
        state.emit(f"{not_cond} = eq {cond_value}, 0")
        state.emit(f"pred_begin {not_cond}")
        emit_region_until(false_label, merge, loop_header, state, seen)
        state.emit("pred_end")
    emit_region_until(merge, stop_label, loop_header, state, seen)


def split_block(lines: list[str]) -> tuple[list[str], tuple[str, ...] | None]:
    if not lines:
        return [], None
    term = terminator(lines[-1])
    if term:
        return lines[:-1], term
    return lines, None


def terminator(line: str) -> tuple[str, ...] | None:
    if line.startswith("ret "):
        parts = line.split()
        if len(parts) >= 3 and parts[1] != "void":
            return ("ret", parts[-1])
        return ("ret",)
    match = BR_COND_RE.match(line)
    if match:
        return ("br_cond", match.group(1), match.group(2), match.group(3))
    match = BR_RE.match(line)
    if match:
        return ("br", match.group(1))
    return None


def reaches(start: str, target: str, blocks: dict[str, list[str]], seen: set[str]) -> bool:
    if start == target:
        return True
    if start in seen or start not in blocks:
        return False
    seen.add(start)
    _, term = split_block(blocks[start])
    if term is None or term[0] == "ret":
        return False
    if term[0] == "br":
        return reaches(term[1], target, blocks, seen)
    return reaches(term[2], target, blocks, seen) or reaches(term[3], target, blocks, seen)


def reaches_avoiding(
    start: str,
    target: str,
    avoid: str,
    blocks: dict[str, list[str]],
    seen: set[str],
) -> bool:
    if start == avoid:
        return False
    if start == target:
        return True
    if start in seen or start not in blocks:
        return False
    seen.add(start)
    _, term = split_block(blocks[start])
    if term is None or term[0] == "ret":
        return False
    if term[0] == "br":
        return reaches_avoiding(term[1], target, avoid, blocks, seen)
    return reaches_avoiding(term[2], target, avoid, blocks, seen) or reaches_avoiding(
        term[3], target, avoid, blocks, seen
    )


def reaches_unconditional(
    start: str,
    target: str,
    avoid: str,
    blocks: dict[str, list[str]],
    seen: set[str],
) -> bool:
    """Return true when start reaches target through only unconditional branches."""
    if start == avoid:
        return False
    if start == target:
        return True
    if start in seen or start not in blocks:
        return False
    seen.add(start)
    _, term = split_block(blocks[start])
    if term is None or term[0] != "br":
        return False
    return reaches_unconditional(term[1], target, avoid, blocks, seen)


def has_backedge_to(label: str, blocks: dict[str, list[str]]) -> bool:
    """Return true when any latch block branches directly back to label."""
    for block_label, lines in blocks.items():
        if block_label == label:
            continue
        _, term = split_block(lines)
        if term is not None and term[0] == "br" and term[1] == label:
            return True
    return False


def common_merge(left: str, right: str, blocks: dict[str, list[str]]) -> str | None:
    if left == right:
        return left
    left_reachable = reachable_labels(left, blocks)
    right_reachable = reachable_labels(right, blocks)
    common = left_reachable & right_reachable
    if not common:
        return None
    for label in blocks:
        if label in common:
            return label
    return None


def reachable_labels(start: str, blocks: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    stack = [start]
    while stack:
        label = stack.pop()
        if label in out or label not in blocks:
            continue
        out.add(label)
        _, term = split_block(blocks[label])
        if term is None or term[0] == "ret":
            continue
        if term[0] == "br":
            stack.append(term[1])
        else:
            stack.extend([term[2], term[3]])
    return out


def lower_instruction(line: str, state: LowerState) -> None:
    assign = ASSIGN_RE.match(line)
    if assign:
        dst, expr = assign.group(1), assign.group(2)
        lower_assignment(dst, expr, state)
        return
    store = STORE_RE.match(line)
    if store:
        _, raw_value, raw_ptr = store.groups()
        lower_store(raw_value, raw_ptr, state)
        return
    if line.startswith("call "):
        lower_call_expr(line, state, None)
        return
    raise LlvmLoweringError(f"unsupported LLVM instruction: {line}")


def lower_assignment(dst: str, expr: str, state: LowerState) -> None:
    alloca = ALLOCA_RE.search(expr)
    if alloca:
        state.slot_types[dst] = llvm_format(alloca.group(1))
        state.slots[dst] = f"%slot{dst[1:]}"
        return

    load = LOAD_RE.match(expr)
    if load:
        fmt = llvm_format(load.group(1))
        raw_ptr = load.group(2)
        metadata = metadata_load(raw_ptr)
        if metadata:
            temp = state.temp()
            state.values[dst] = temp
            state.value_types[temp] = "i32"
            state.emit(f"{temp} = {metadata}")
            return
        ptr = pointer_value(raw_ptr, state)
        if ptr in state.slots:
            current = state.slots[ptr]
            state.values[dst] = current
            state.value_types[current] = state.slot_types.get(ptr, fmt)
            return
        if ptr in state.pointers:
            addr, ptr_fmt = state.pointers[ptr]
            temp = state.temp()
            state.values[dst] = temp
            fmt = ptr_fmt if ptr_fmt != "ptr" else fmt
            state.value_types[temp] = fmt
            state.emit(f"{temp} = {typed_ir_op('load_global', fmt)} {addr}")
            return
        raise LlvmLoweringError(f"unsupported load pointer: {raw_ptr}")

    gep = GEP_RE.match(expr)
    if gep:
        elem_type, raw_base, raw_indices = gep.groups()
        base = value(raw_base, state)
        indices = split_operands(raw_indices)
        if not indices:
            raise LlvmLoweringError(f"getelementptr without index: {expr}")
        index = value(indices[-1].split()[-1], state)
        fmt = pointer_element_format(raw_base, llvm_format(elem_type), state)
        addr = lower_address(base, index, fmt, state)
        state.pointers[dst] = (addr, fmt)
        state.value_types[addr] = "i32"
        return

    call = CALL_RE.search(expr)
    if call:
        lower_call_expr(expr, state, dst)
        return

    if expr.startswith(("sext ", "zext ", "trunc ", "bitcast ")):
        state.values[dst] = value(split_operands(expr.split(" to ", 1)[0])[-1].split()[-1], state)
        return

    if expr.startswith(("fptosi ", "fptoui ")):
        src = split_operands(expr.split(" to ", 1)[0])[-1].split()[-1]
        temp = state.temp()
        state.values[dst] = temp
        state.value_types[temp] = "i32"
        state.emit(f"{temp} = ftoi.fp32 {value(src, state)}")
        return

    if expr.startswith(("sitofp ", "uitofp ")):
        src = split_operands(expr.split(" to ", 1)[0])[-1].split()[-1]
        temp = state.temp()
        state.values[dst] = temp
        state.value_types[temp] = "fp32"
        state.emit(f"{temp} = itof.i32 {value(src, state)}")
        return

    if expr.startswith("icmp "):
        pred, lhs, rhs = parse_typed_binary(expr, "icmp")
        temp = emit_binary(ICMP_OPS[pred], lhs, rhs, "i32", state)
        state.values[dst] = temp
        return

    if expr.startswith("fcmp "):
        pred, lhs, rhs = parse_typed_binary(expr, "fcmp")
        temp = emit_binary(FCMP_OPS[pred], lhs, rhs, "fp32", state, result_type="i32")
        state.values[dst] = temp
        return

    op = expr.split(None, 1)[0]
    if op in INT_OPS or op in FLOAT_OPS:
        _, lhs, rhs = parse_typed_binary(expr, op)
        fmt = "fp32" if op in FLOAT_OPS else merged_type(lhs, rhs, state)
        temp = emit_binary((INT_OPS | FLOAT_OPS)[op], lhs, rhs, fmt, state)
        state.values[dst] = temp
        return

    raise LlvmLoweringError(f"unsupported LLVM assignment: {dst} = {expr}")


def lower_store(raw_value: str, raw_ptr: str, state: LowerState) -> None:
    ptr = pointer_value(raw_ptr, state)
    stored = value(raw_value.split()[-1], state)
    if ptr in state.slots:
        slot_name = state.slots[ptr]
        state.slots[ptr] = slot_name
        state.value_types[slot_name] = state.value_types.get(stored, state.slot_types.get(ptr, "i32"))
        source_token = raw_value.split()[-1]
        if (
            source_token in state.function.params
            or (stored.startswith("%") and state.value_types.get(stored) == "ptr")
        ):
            state.slots[ptr] = stored
        else:
            state.emit(f"{slot_name} = mov {stored}")
        return
    if ptr in state.pointers:
        addr, fmt = state.pointers[ptr]
        state.emit(f"{typed_ir_op('store_global', fmt)} {addr}, {stored}")
        return
    raise LlvmLoweringError(f"unsupported store pointer: {raw_ptr}")


def lower_call_expr(expr: str, state: LowerState, dst: str | None) -> None:
    call = CALL_RE.search(expr)
    if not call:
        raise LlvmLoweringError(f"unsupported call: {expr}")
    callee, args_text = call.groups()
    args = [arg.split()[-1] for arg in split_operands(args_text)]
    if (
        "minigpu_as_u32" in callee
        or "minigpu_as_f32" in callee
        or "__float_as_uint" in callee
        or "__uint_as_float" in callee
    ):
        if len(args) != 1:
            raise LlvmLoweringError(f"{callee} expects one argument")
        if dst is not None:
            state.values[dst] = value(args[0], state)
            if "minigpu_as_u32" in callee or "__float_as_uint" in callee:
                state.value_types[dst] = "i32"
            else:
                state.value_types[dst] = "fp32"
        return
    if "__syncthreads" in callee:
        state.emit("barrier")
        return
    if callee in state.shared_helpers:
        if dst is not None:
            lowered_args = [value(arg, state) for arg in args]
            result_fmt = state.shared_helper_returns.get(callee, "fp32")
            state.value_types[dst] = result_fmt
            state.emit(f"{dst} = call.{result_fmt} {callee}, {', '.join(lowered_args)}")
        return
    raise LlvmLoweringError(f"unsupported call to {callee}")


def emit_binary(
    op: str,
    lhs: str,
    rhs: str,
    fmt: str,
    state: LowerState,
    *,
    result_type: str | None = None,
) -> str:
    left = value(lhs, state)
    right = value(rhs, state)
    temp = state.temp()
    state.value_types[temp] = result_type or fmt
    state.emit(f"{temp} = {typed_ir_op(op, fmt)} {left}, {right}")
    return temp


def lower_address(base: str, index: str, fmt: str, state: LowerState) -> str:
    if is_packed_format(fmt):
        base_shift = const(2, state)
        base_bytes = emit_binary("shl", base, base_shift, "i32", state)
        elem_shift = const(packed_byte_shift(fmt), state)
        elem_bytes = emit_binary("shl", index, elem_shift, "i32", state)
        return emit_binary("add", base_bytes, elem_bytes, "i32", state)
    return emit_binary("add", base, index, "i32", state)


def const(number: int, state: LowerState) -> str:
    temp = state.temp()
    state.value_types[temp] = "i32"
    state.emit(f"{temp} = const {number}")
    return temp


def value(token: str, state: LowerState) -> str:
    token = token.strip()
    if token in state.values:
        return state.values[token]
    if token in state.value_types:
        return token
    if token in state.slots:
        return state.slots[token]
    if re.match(r"^-?\d+$", token):
        temp = state.temp()
        state.value_types[temp] = "i32"
        state.emit(f"{temp} = const {token}")
        return temp
    if is_float_literal(token):
        temp = state.temp()
        state.value_types[temp] = "fp32"
        state.emit(f"{temp} = const {float_literal_bits(token)}")
        return temp
    raise LlvmLoweringError(f"use of unknown LLVM value: {token}")


def is_float_literal(token: str) -> bool:
    """Return true for the decimal/scientific float constants Clang emits."""
    return bool(
        re.match(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+\.)(?:[eE][+-]?\d+)?$", token)
        or re.match(r"^0x[0-9A-Fa-f]{16}$", token)
    )


def float_literal_bits(token: str) -> int:
    """Return the raw IEEE-754 single-precision bits for a float literal."""
    if token.startswith("0x"):
        value = struct.unpack(">d", int(token, 16).to_bytes(8, "big"))[0]
    else:
        value = float(token)
    return struct.unpack("<I", struct.pack("<f", value))[0]


def pointer_value(raw_ptr: str, state: LowerState) -> str:
    raw_ptr = raw_ptr.strip()
    if raw_ptr in state.slots or raw_ptr in state.pointers:
        return raw_ptr
    if raw_ptr in state.values:
        return state.values[raw_ptr]
    return raw_ptr


def metadata_load(raw_ptr: str) -> str | None:
    if "@threadIdx" in raw_ptr:
        return "thread_idx"
    if "@blockIdx" in raw_ptr:
        return "block_idx"
    if "@blockDim" in raw_ptr:
        return "block_dim"
    if "@gridDim" in raw_ptr:
        return "grid_dim"
    return None


def pointer_element_format(raw_base: str, fallback: str, state: LowerState) -> str:
    base = value(raw_base, state)
    if base.startswith("%"):
        name = base[1:]
        if name in state.meta.arg_names:
            return state.meta.arg_formats[state.meta.arg_names.index(name)]
    return fallback


def parse_typed_binary(expr: str, op: str) -> tuple[str, str, str]:
    rest = expr[len(op):].strip()
    parts = split_operands(rest)
    if len(parts) != 2:
        raise LlvmLoweringError(f"expected binary operands in: {expr}")
    head = parts[0].split()
    while head and head[0] in {"nsw", "nuw", "exact", "contract", "fast"}:
        head.pop(0)
    pred_or_lhs = head[-1] if op not in {"icmp", "fcmp"} else head[0]
    lhs = head[-1]
    rhs = parts[1].split()[-1]
    return pred_or_lhs, lhs, rhs


def merged_type(lhs: str, rhs: str, state: LowerState) -> str:
    left = token_type(lhs, state)
    right = token_type(rhs, state)
    return merge_value_formats(left, right)


def token_type(token: str, state: LowerState) -> str:
    token = token.strip()
    if token in state.value_types:
        return state.value_types[token]
    if token in state.values:
        return state.value_types.get(state.values[token], "i32")
    if token in state.slots:
        return state.value_types.get(state.slots[token], state.slot_types.get(token, "i32"))
    return "i32"


def llvm_format(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("void"):
        return "void"
    if stripped.startswith("float"):
        return "fp32"
    if stripped.startswith("half"):
        return "fp16"
    if stripped.startswith("i16"):
        return "i16"
    if stripped.startswith("i8"):
        return "i8"
    return "i32" if stripped.startswith("i") else "ptr"


def split_operands(text: str) -> list[str]:
    operands: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            operands.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        operands.append(tail)
    return operands


def type_text(node: dict[str, Any]) -> str:
    type_info = node.get("type")
    if isinstance(type_info, dict):
        return str(type_info.get("qualType", ""))
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lower LLVM IR to Mini-GPU IR.")
    parser.add_argument("llvm_ir", type=Path)
    parser.add_argument("ast_json", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.loads(args.ast_json.read_text(encoding="utf-8"))
    ir = llvm_to_ir(args.llvm_ir.read_text(encoding="utf-8"), document)
    if args.output:
        args.output.write_text(ir + "\n", encoding="utf-8")
    else:
        print(ir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
