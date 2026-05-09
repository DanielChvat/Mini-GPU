"""PrivateUse1 registration helpers for the Mini-GPU PyTorch backend."""

from __future__ import annotations

import importlib
import sys
from typing import Iterable

import torch

BACKEND_NAME = "minigpu"
EXTENSION_MODULE = "torch_mini_gpu.minigpu_torch"

_initialized = False
_extension = None
_print_hook_installed = False
_original_tensor_str = None


def _load_extension():
    """Import and cache the native Mini-GPU PyTorch extension."""
    global _extension

    if _extension is None:
        _extension = importlib.import_module(EXTENSION_MODULE)
    return _extension


def init() -> None:
    """Register the PrivateUse1 backend name and load the C++ extension."""
    global _initialized

    if _initialized:
        return

    # PyTorch allows the PrivateUse1 backend to be renamed once per process.
    torch.utils.rename_privateuse1_backend(BACKEND_NAME)

    ext = _load_extension()
    ext.init()

    # PyTorch expects a module object at torch.<backend_name> for device helpers.
    torch._register_device_module(BACKEND_NAME, sys.modules[__name__])

    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
        unsupported_dtype=_unsupported_storage_dtypes(),
    )
    _install_print_hook()

    _initialized = True


def is_built() -> bool:
    """Return true when the Mini-GPU PyTorch extension can be imported."""
    try:
        _load_extension()
    except ImportError:
        return False
    return True


def is_available() -> bool:
    """Return true when the extension is built and hardware is available."""
    if not is_built():
        return False
    return bool(_load_extension().is_available())


def connect(port: str, baud: int = 115200, memory_size: int = 65536) -> None:
    """Open the board transport used by tensor.to('minigpu')."""
    init()
    _load_extension().connect(str(port), int(baud), int(memory_size))


def disconnect() -> None:
    """Close the board transport used by Mini-GPU tensors."""
    if is_built():
        _load_extension().disconnect()


def device_count() -> int:
    """Return the number of Mini-GPU devices visible to the runtime."""
    return int(_load_extension().device_count())


def get_device() -> int:
    """Return the active Mini-GPU device index for this process."""
    return int(_load_extension().get_device())


def current_device() -> int:
    """Return the active Mini-GPU device index for PyTorch device helpers."""
    return get_device()


def set_device(index: int) -> None:
    """Set the active Mini-GPU device index for later PyTorch operations."""
    _load_extension().set_device(int(index))


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


def _install_print_hook() -> None:
    """Format Mini-GPU tensors through a temporary CPU copy."""
    global _print_hook_installed, _original_tensor_str

    if _print_hook_installed:
        return

    import torch._tensor_str as tensor_str

    _original_tensor_str = tensor_str._str

    def _minigpu_aware_str(tensor, *, tensor_contents=None):
        if tensor_contents is None and getattr(tensor, "device", None).type == BACKEND_NAME:
            prefix = "tensor(" if type(tensor) is torch.Tensor else f"{type(tensor).__name__}("
            cpu_tensor = tensor.to("cpu")
            tensor_contents = tensor_str._tensor_str(cpu_tensor, len(prefix))
        return _original_tensor_str(tensor, tensor_contents=tensor_contents)

    tensor_str._str = _minigpu_aware_str
    _print_hook_installed = True
