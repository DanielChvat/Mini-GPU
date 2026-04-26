from enum import IntEnum

class Error(IntEnum):
    OK: int
    BAD_ARG: int
    CRC: int
    NO_SOF: int
    TIMEOUT: int
    SHORT: int
    IO: int
    PAYLOAD_SIZE: int
    OPEN: int
    FPGA: int

def strerror(err: int | Error) -> str: ...

class Device:
    def __init__(
        self,
        device_name: str,
        baud: int,
        timeout_ms: int = 1000,
    ) -> None: ...

    def close(self) -> None: ...

    def is_open(self) -> bool: ...

    def send_raw(self, data: bytes) -> None: ...

    def recv_raw(self, n: int) -> bytes: ...