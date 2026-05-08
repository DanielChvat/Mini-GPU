"""Python entry point for the Mini-GPU PyTorch backend."""

from .backend import (
    BACKEND_NAME,
    connect,
    device_count,
    disconnect,
    get_device,
    init,
    is_available,
    is_built,
    set_device,
)

__all__ = [
    "BACKEND_NAME",
    "connect",
    "device_count",
    "disconnect",
    "get_device",
    "init",
    "is_available",
    "is_built",
    "set_device",
]
