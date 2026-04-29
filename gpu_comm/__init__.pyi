from enum import IntEnum
from typing import TypedDict

MAX_PAYLOAD: int


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


class Command(IntEnum):
    WRITE_DATA: int
    READ_DATA: int
    WRITE_PROGRAM: int
    LAUNCH: int
    READ_STATUS: int
    WRITE_HASH: int
    VALIDATE: int


class ParsedPacket(TypedDict):
    cmd: int
    addr: int
    len: int
    payload: bytes


def strerror(err: int | Error) -> str: ...

def build_packet(
    cmd: int | Command,
    addr: int,
    payload: bytes = b"",
) -> bytes: ...

def parse_packet(data: bytes) -> ParsedPacket: ...


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

    def write_data(self, addr: int, data: bytes) -> None: ...

    def read_data(self, addr: int, n: int) -> bytes: ...
