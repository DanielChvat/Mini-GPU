#!/usr/bin/env python3
"""Lower compact CUDA AST JSON into Mini-GPU IR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ARITHMETIC_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "&": "and",
    "|": "or",
    "^": "xor",
    "<<": "shl",
    ">>": "shr",
}

COMPARISON_OPS = {
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
}

COMPOUND_ASSIGN_OPS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}


class LoweringError(Exception):
    """Raised when the CUDA AST uses an unsupported construct."""

    pass


@dataclass
class KernelState:
    name: str
    lines: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    args: dict[str, str] = field(default_factory=dict)
    temp_index: int = 0
    label_index: int = 0

    def emit(self, text: str = "") -> None:
        """Append one IR line."""
        self.lines.append(text)

    def temp(self) -> str:
        """Allocate a temporary SSA-like IR value."""
        self.temp_index += 1
        return f"%t{self.temp_index}"

    def label(self, prefix: str) -> str:
        """Allocate a unique label name inside the kernel."""
        self.label_index += 1
        return f"{prefix}_{self.label_index}"


class MiniGpuIrLowerer:
    def lower_document(self, document: dict[str, Any]) -> str:
        """Lower all kernels in a compact AST document."""
        kernels = document.get("kernels")
        if not isinstance(kernels, list):
            raise LoweringError("input JSON must contain a 'kernels' list")

        chunks = [self.lower_kernel(kernel) for kernel in kernels]
        return "\n\n".join(chunks)

    def lower_kernel(self, kernel: dict[str, Any]) -> str:
        """Lower one `__global__` function."""
        state = KernelState(name=kernel.get("name", "<anonymous>"))
        state.emit(f"kernel {state.name}")

        body = None
        for child in children(kernel):
            if child.get("kind") == "ParmVarDecl":
                name = child.get("name")
                if not name:
                    raise LoweringError(f"kernel {state.name} has an unnamed parameter")
                value = f"%{name}"
                state.args[name] = value
                state.values[name] = value
                state.emit(f"{value} = arg {name}")
            elif child.get("kind") == "CompoundStmt":
                body = child

        if body is None:
            raise LoweringError(f"kernel {state.name} has no body")

        if state.args:
            state.emit()

        self.lower_stmt(body, state)

        if not state.lines or state.lines[-1] != "return":
            state.emit("return")
        return "\n".join(state.lines)

    def lower_stmt(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower a statement node."""
        kind = node.get("kind")

        if kind == "CompoundStmt":
            for child in children(node):
                self.lower_stmt(child, state)
            return

        if kind == "DeclStmt":
            for child in children(node):
                if child.get("kind") == "VarDecl":
                    self.lower_var_decl(child, state)
                else:
                    self.unsupported(child, "declaration")
            return

        if kind in {"BinaryOperator", "CompoundAssignOperator"}:
            self.lower_binary_stmt(node, state)
            return

        if kind == "IfStmt":
            self.lower_if(node, state)
            return

        if kind == "ForStmt":
            self.lower_for(node, state)
            return

        if kind == "ReturnStmt":
            state.emit("return")
            return

        if kind == "UnaryOperator":
            self.lower_unary_stmt(node, state)
            return

        if kind == "CallExpr":
            self.lower_call(node, state)
            return

        if kind in {"NullStmt", "CUDAGlobalAttr"}:
            return

        self.unsupported(node, "statement")

    def lower_var_decl(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower a local scalar declaration."""
        name = node.get("name")
        if not name:
            raise LoweringError("encountered unnamed local variable")

        init = first_expr_child(node)
        if init is None:
            value = state.temp()
            state.emit(f"{value} = const 0")
        else:
            value = self.lower_expr(init, state)
        state.values[name] = value
        state.emit(f"%{name} = mov {value}")
        state.values[name] = f"%{name}"

    def lower_binary_stmt(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower assignment and compound-assignment statements."""
        opcode = node.get("opcode")
        operands = children(node)
        if len(operands) != 2:
            self.unsupported(node, "binary statement with unexpected arity")

        lhs, rhs = operands
        if opcode == "=":
            self.assign(lhs, rhs, state)
            return

        if opcode in COMPOUND_ASSIGN_OPS:
            current = self.lower_expr(lhs, state)
            update = self.lower_binary_expr(COMPOUND_ASSIGN_OPS[opcode], current, rhs, state)
            self.assign_value(lhs, update, state)
            return

        self.lower_expr(node, state)

    def lower_if(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower `if` using predicate-mask IR."""
        parts = children(node)
        if len(parts) < 2:
            self.unsupported(node, "if statement")

        cond = self.lower_expr(parts[0], state)
        state.emit(f"pred_begin {cond}")
        self.lower_stmt(parts[1], state)
        state.emit("pred_end")

        if len(parts) > 2:
            not_cond = state.temp()
            state.emit(f"{not_cond} = eq {cond}, 0")
            state.emit(f"pred_begin {not_cond}")
            self.lower_stmt(parts[2], state)
            state.emit("pred_end")

    def lower_for(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower a simple C-style `for` loop."""
        parts = [part for part in children(node) if part.get("kind")]
        if len(parts) != 4:
            self.unsupported(node, "for statement")

        init, cond, inc, body = parts
        loop_label = state.label("for_begin")
        end_label = state.label("for_end")

        if init.get("kind") != "NullStmt":
            self.lower_stmt(init, state)

        state.emit(f"label {loop_label}")
        cond_value = self.lower_expr(cond, state)
        state.emit(f"branch_zero {cond_value}, {end_label}")
        self.lower_stmt(body, state)
        if inc.get("kind") != "NullStmt":
            self.lower_stmt(inc, state)
        state.emit(f"branch {loop_label}")
        state.emit(f"label {end_label}")

    def lower_unary_stmt(self, node: dict[str, Any], state: KernelState) -> None:
        """Lower increment and decrement statements."""
        opcode = node.get("opcode")
        operands = children(node)
        if len(operands) != 1:
            self.unsupported(node, "unary statement with unexpected arity")

        target = operands[0]
        current = self.lower_expr(target, state)
        one = state.temp()
        state.emit(f"{one} = const 1")
        if opcode in {"++", "post++"}:
            updated = self.lower_binary_value("add", current, one, state)
        elif opcode in {"--", "post--"}:
            updated = self.lower_binary_value("sub", current, one, state)
        else:
            self.unsupported(node, f"unary statement opcode {opcode!r}")
        self.assign_value(target, updated, state)

    def lower_call(self, node: dict[str, Any], state: KernelState) -> str | None:
        """Lower supported CUDA builtins."""
        callee = call_name(node)
        if callee == "__syncthreads":
            state.emit("barrier")
            return None
        self.unsupported(node, f"call to {callee or '<unknown>'}")

    def assign(self, lhs: dict[str, Any], rhs: dict[str, Any], state: KernelState) -> None:
        """Lower an assignment from an expression."""
        value = self.lower_expr(rhs, state)
        self.assign_value(lhs, value, state)

    def assign_value(self, lhs: dict[str, Any], value: str, state: KernelState) -> None:
        """Store an already-lowered value into a variable or array element."""
        lhs = unwrap(lhs)
        kind = lhs.get("kind")

        if kind == "DeclRefExpr":
            name = referenced_name(lhs)
            if not name:
                self.unsupported(lhs, "assignment target")
            state.values[name] = value
            state.emit(f"%{name} = mov {value}")
            state.values[name] = f"%{name}"
            return

        if kind == "ArraySubscriptExpr":
            base, index = array_parts(lhs)
            base_value = self.lower_expr(base, state)
            index_value = self.lower_expr(index, state)
            addr = state.temp()
            state.emit(f"{addr} = add {base_value}, {index_value}")
            state.emit(f"store_global {addr}, {value}")
            return

        self.unsupported(lhs, "assignment target")

    def lower_expr(self, node: dict[str, Any], state: KernelState) -> str:
        """Lower an expression and return its IR value name."""
        node = unwrap(node)
        kind = node.get("kind")

        if kind == "DeclRefExpr":
            name = referenced_name(node)
            if name in state.values:
                return state.values[name]
            if name in state.args:
                return state.args[name]
            self.unsupported(node, f"reference to unknown value {name!r}")

        if kind == "IntegerLiteral":
            value = literal_value(node)
            temp = state.temp()
            state.emit(f"{temp} = const {value}")
            return temp

        if kind == "MemberExpr":
            special = cuda_metadata_name(node)
            if special:
                temp = state.temp()
                state.emit(f"{temp} = {special}")
                return temp
            self.unsupported(node, "member expression")

        if kind in {"BinaryOperator", "CompoundAssignOperator"}:
            if is_global_tid_expr(node):
                temp = state.temp()
                state.emit(f"{temp} = global_tid")
                return temp

            opcode = node.get("opcode")
            operands = children(node)
            if len(operands) != 2:
                self.unsupported(node, "binary expression with unexpected arity")
            if opcode in ARITHMETIC_OPS:
                return self.lower_binary_expr(opcode, operands[0], operands[1], state)
            if opcode in COMPARISON_OPS:
                return self.lower_binary_expr(opcode, operands[0], operands[1], state)
            if opcode == "=":
                self.assign(operands[0], operands[1], state)
                return self.lower_expr(operands[0], state)
            self.unsupported(node, f"binary expression opcode {opcode!r}")

        if kind == "ArraySubscriptExpr":
            base, index = array_parts(node)
            base_value = self.lower_expr(base, state)
            index_value = self.lower_expr(index, state)
            addr = state.temp()
            value = state.temp()
            state.emit(f"{addr} = add {base_value}, {index_value}")
            state.emit(f"{value} = load_global {addr}")
            return value

        if kind == "UnaryOperator":
            opcode = node.get("opcode")
            operands = children(node)
            if len(operands) != 1:
                self.unsupported(node, "unary expression with unexpected arity")
            if opcode == "-":
                zero = state.temp()
                state.emit(f"{zero} = const 0")
                return self.lower_binary_expr("-", zero, operands[0], state)
            if opcode == "~":
                value = self.lower_expr(operands[0], state)
                temp = state.temp()
                state.emit(f"{temp} = not {value}")
                return temp
            if opcode == "!":
                value = self.lower_expr(operands[0], state)
                temp = state.temp()
                state.emit(f"{temp} = eq {value}, 0")
                return temp
            if opcode in {"++", "post++", "--", "post--"}:
                self.lower_unary_stmt(node, state)
                return self.lower_expr(operands[0], state)
            self.unsupported(node, f"unary expression opcode {opcode!r}")

        if kind == "CallExpr":
            result = self.lower_call(node, state)
            if result is None:
                self.unsupported(node, "void call used as expression")
            return result

        self.unsupported(node, "expression")

    def lower_binary_expr(
        self,
        opcode: str,
        lhs: dict[str, Any] | str,
        rhs: dict[str, Any] | str,
        state: KernelState,
    ) -> str:
        """Lower a binary expression from AST opcode to IR opcode."""
        if isinstance(lhs, str):
            lhs_value = lhs
        else:
            lhs_value = self.lower_expr(lhs, state)

        if isinstance(rhs, str):
            rhs_value = rhs
        else:
            rhs_value = self.lower_expr(rhs, state)

        ir_opcode = ARITHMETIC_OPS.get(opcode) or COMPARISON_OPS.get(opcode)
        if ir_opcode is None:
            raise LoweringError(f"unsupported binary opcode: {opcode}")
        return self.lower_binary_value(ir_opcode, lhs_value, rhs_value, state)

    def lower_binary_value(self, ir_opcode: str, lhs: str, rhs: str, state: KernelState) -> str:
        """Emit a binary IR operation on existing IR values."""
        temp = state.temp()
        state.emit(f"{temp} = {ir_opcode} {lhs}, {rhs}")
        return temp

    def unsupported(self, node: dict[str, Any], context: str) -> None:
        """Raise a source-aware unsupported-feature error."""
        loc = node.get("loc", {})
        line = loc.get("line")
        col = loc.get("col")
        where = f" at {line}:{col}" if line is not None and col is not None else ""
        raise LoweringError(f"unsupported {context}: {node.get('kind')}{where}")


def children(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return AST child nodes."""
    return [child for child in node.get("inner", []) if isinstance(child, dict)]


def unwrap(node: dict[str, Any]) -> dict[str, Any]:
    """Skip Clang wrapper nodes that do not affect Mini-GPU semantics."""
    while node.get("kind") in {
        "ImplicitCastExpr",
        "CStyleCastExpr",
        "ParenExpr",
        "MaterializeTemporaryExpr",
    }:
        inner = children(node)
        if len(inner) != 1:
            return node
        node = inner[0]
    return node


def first_expr_child(node: dict[str, Any]) -> dict[str, Any] | None:
    """Find the initializer/expression child for a declaration."""
    for child in children(node):
        if child.get("kind") not in {"CUDAGlobalAttr"}:
            return child
    return None


def referenced_name(node: dict[str, Any]) -> str | None:
    """Return the variable/function name referenced by an AST node."""
    if "referencedDecl" in node:
        return node["referencedDecl"].get("name")
    return node.get("name")


def literal_value(node: dict[str, Any]) -> int:
    """Return an integer literal value."""
    if "value" in node:
        return int(node["value"])
    if "inner" in node:
        raise LoweringError("integer literal has unexpected child nodes")
    raise LoweringError("integer literal is missing a value")


def array_parts(node: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split `base[index]` into base and index expressions."""
    parts = children(node)
    if len(parts) != 2:
        raise LoweringError("array subscript expression has unexpected arity")
    return parts[0], parts[1]


def cuda_metadata_name(node: dict[str, Any]) -> str | None:
    """Map CUDA metadata fields like `threadIdx.x` to IR op names."""
    member = node.get("name")
    if member not in {"x", "y", "z"}:
        return None

    base_children = children(node)
    if len(base_children) != 1:
        return None

    base = unwrap(base_children[0])
    if base.get("kind") != "DeclRefExpr":
        return None

    base_name = referenced_name(base)
    if member != "x":
        raise LoweringError(f"only x-dimension CUDA metadata is supported, got {base_name}.{member}")

    return {
        "threadIdx": "thread_idx",
        "blockIdx": "block_idx",
        "blockDim": "block_dim",
        "gridDim": "grid_dim",
    }.get(base_name)


def is_cuda_metadata_expr(node: dict[str, Any], base_name: str) -> bool:
    """Check for a specific CUDA metadata expression."""
    node = unwrap(node)
    return node.get("kind") == "MemberExpr" and cuda_metadata_base(node) == base_name


def cuda_metadata_base(node: dict[str, Any]) -> str | None:
    """Return `threadIdx`, `blockIdx`, etc. for a `.x` member expression."""
    if node.get("name") != "x":
        return None

    base_children = children(node)
    if len(base_children) != 1:
        return None

    base = unwrap(base_children[0])
    if base.get("kind") != "DeclRefExpr":
        return None
    return referenced_name(base)


def is_global_tid_expr(node: dict[str, Any]) -> bool:
    """Detect `blockIdx.x * blockDim.x + threadIdx.x`."""
    node = unwrap(node)
    if node.get("kind") != "BinaryOperator" or node.get("opcode") != "+":
        return False

    operands = children(node)
    if len(operands) != 2:
        return False

    return (
        is_block_linear_base(operands[0])
        and is_cuda_metadata_expr(operands[1], "threadIdx")
    ) or (
        is_cuda_metadata_expr(operands[0], "threadIdx")
        and is_block_linear_base(operands[1])
    )


def is_block_linear_base(node: dict[str, Any]) -> bool:
    """Detect `blockIdx.x * blockDim.x`."""
    node = unwrap(node)
    if node.get("kind") != "BinaryOperator" or node.get("opcode") != "*":
        return False

    operands = children(node)
    if len(operands) != 2:
        return False

    return (
        is_cuda_metadata_expr(operands[0], "blockIdx")
        and is_cuda_metadata_expr(operands[1], "blockDim")
    ) or (
        is_cuda_metadata_expr(operands[0], "blockDim")
        and is_cuda_metadata_expr(operands[1], "blockIdx")
    )


def call_name(node: dict[str, Any]) -> str | None:
    """Return the callee name for a call expression."""
    for child in children(node):
        child = unwrap(child)
        if child.get("kind") == "DeclRefExpr":
            return referenced_name(child)
    return None


def ast_to_ir(document: dict[str, Any]) -> str:
    """Lower compact AST JSON data to Mini-GPU IR text."""
    return MiniGpuIrLowerer().lower_document(document)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lower compact CUDA AST JSON to Mini-GPU IR.")
    parser.add_argument("ast_json", type=Path, help="JSON file emitted by get_cuda_ast.py")
    parser.add_argument("-o", "--output", type=Path, help="Write IR to this file instead of stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.ast_json.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    ir = ast_to_ir(document)
    if args.output:
        args.output.write_text(ir + "\n", encoding="utf-8")
    else:
        print(ir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
