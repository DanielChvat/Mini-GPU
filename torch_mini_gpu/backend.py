"""PrivateUse1 registration helpers for the Mini-GPU PyTorch backend."""

from __future__ import annotations

import importlib
from typing import Iterable

import torch

BACKEND_NAME = "minigpu"

_initialized = False
_extension = None


def init() -> None:
    """Register the PrivateUse1 backend name and load the C++ extension."""
    global _initialized, _extension

    if _initialized:
        return

    # PyTorch allows the PrivateUse1 backend to be renamed once per process.
    torch.utils.rename_privateuse1_backend(BACKEND_NAME)
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
        unsupported_dtype=_unsupported_storage_dtypes(),
    )

    # This import will succeed once the torch extension is built and installed.
    _extension = importlib.import_module("torch_mini_gpu._C")
    _extension.init()
    _initialized = True


def is_built() -> bool:
    """Return true when the Mini-GPU PyTorch extension can be imported."""
    try:
        importlib.import_module("torch_mini_gpu._C")
    except ImportError:
        return False
    return True


def is_available() -> bool:
    """Return true when the extension is built and hardware is available."""
    if not is_built():
        return False
    ext = importlib.import_module("torch_mini_gpu._C")
    return bool(ext.is_available())


def device_count() -> int:
    """Return the number of Mini-GPU devices visible to the runtime."""
    init()
    return int(_extension.device_count())


def get_device() -> int:
    """Return the active Mini-GPU device index for this process."""
    init()
    return int(_extension.get_device())


def set_device(index: int) -> None:
    """Set the active Mini-GPU device index for later PyTorch operations."""
    init()
    _extension.set_device(int(index))


def _unsupported_storage_dtypes() -> Iterable[torch.dtype]:
    """List dtypes that should not get generated storage helpers yet."""
    return [
        torch.bool,
        torch.complex64,
        torch.complex128,
        torch.quint8,
        torch.qint8,
        torch.qint32,
    ]
