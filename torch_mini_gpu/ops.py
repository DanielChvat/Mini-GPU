"""User-facing PyTorch operation helpers for Mini-GPU experiments."""

from __future__ import annotations

import torch

from .backend import BACKEND_NAME, init


def to_minigpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor onto the Mini-GPU PrivateUse1 device."""
    init()
    return tensor.to(device=f"{BACKEND_NAME}:0")


def empty(shape, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """Create an empty Mini-GPU tensor once the C++ empty stub is implemented."""
    init()
    return torch.empty(shape, dtype=dtype, device=device or f"{BACKEND_NAME}:0")


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Call the Mini-GPU vector-add custom op stub."""
    init()
    return torch.ops.minigpu.vector_add(a, b)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Call the Mini-GPU matmul custom op stub."""
    init()
    return torch.ops.minigpu.matmul(a, b)


def relu(a: torch.Tensor) -> torch.Tensor:
    """Call the Mini-GPU ReLU custom op stub."""
    init()
    return torch.ops.minigpu.relu(a)
