import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu_comm


def make_payload(n):
    return bytes(i & 0xFF for i in range(n))


def parse_int(value):
    return int(value, 0)


def parse_hex_bytes(value):
    cleaned = value.replace("_", "").replace(" ", "").replace(",", "")
    if len(cleaned) % 2 != 0:
        raise ValueError("hex payload must contain an even number of hex digits")
    return bytes.fromhex(cleaned)


def print_packet(packet):
    print("packet:", packet.hex(" "))
    print("packet length:", len(packet))


def recv_packet(dev):
    header = bytes(dev.recv_raw(6))
    if len(header) != 6 or header[0] != 0xAA:
        raise RuntimeError(f"bad response header: {header.hex(' ')}")

    payload_len = (header[4] << 8) | header[5]
    rest = bytes(dev.recv_raw(payload_len + 1))
    packet = header + rest
    parsed = gpu_comm.parse_packet(packet)
    return parsed, packet


def print_hex_dump(data, base_addr=0, width=16):
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        print(f"0x{base_addr + offset:04x}: {chunk.hex(' ')}")


def rtl_read_chunk(dev, addr, length, verbose=False):
    payload = bytes(length)
    packet = gpu_comm.build_packet(gpu_comm.Command.READ_DATA, addr, payload)

    if verbose:
        print_packet(packet)

    dev.send_raw(packet)
    parsed, response = recv_packet(dev)

    if verbose:
        print("response packet:", response.hex(" "))
        print(f"response cmd=0x{parsed['cmd']:02x} addr=0x{parsed['addr']:04x} len={parsed['len']}")

    if parsed["cmd"] != int(gpu_comm.Command.READ_DATA):
        raise RuntimeError(f"unexpected response cmd: 0x{parsed['cmd']:02x}")
    if parsed["addr"] != addr:
        raise RuntimeError(f"unexpected response addr: 0x{parsed['addr']:04x}")
    if parsed["len"] != length:
        raise RuntimeError(f"unexpected response len: {parsed['len']}")

    return bytes(parsed["payload"])


def open_device(port, baud, timeout_ms):
    dev = gpu_comm.Device(port, baud, timeout_ms)
    print("opened:", dev.is_open())
    return dev


def send_one_packet(args):
    payload = parse_hex_bytes(args.payload)
    packet = gpu_comm.build_packet(gpu_comm.Command.WRITE_DATA, args.addr, payload)

    print_packet(packet)

    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        dev.send_raw(packet)
        print("sent one raw WRITE_DATA packet")
    finally:
        dev.close()
        print("closed")


def send_write(args):
    payload = parse_hex_bytes(args.payload)
    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        dev.write_data(args.addr, payload)
        print(f"write_data acked: addr=0x{args.addr:04x} len={len(payload)}")
    finally:
        dev.close()
        print("closed")


def send_split_write(args):
    max_payload = gpu_comm.MAX_PAYLOAD
    payload = make_payload(max_payload + args.extra_bytes)

    print("max payload:", max_payload)
    print("payload length:", len(payload))
    print("expected packets:", (len(payload) + max_payload - 1) // max_payload)

    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        dev.write_data(args.addr, payload)
        print(f"split write_data acked: addr=0x{args.addr:04x} len={len(payload)}")
    finally:
        dev.close()
        print("closed")


def send_read(args):
    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        data = dev.read_data(args.addr, args.length)
        print(f"read_data returned: addr=0x{args.addr:04x} len={args.length}")
        print(bytes(data).hex(" "))
    finally:
        dev.close()
        print("closed")


def send_rtl_read(args):
    if args.length < 0:
        raise ValueError("length must be non-negative")

    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        out = bytearray()
        offset = 0
        while offset < args.length:
            remaining = args.length - offset
            chunk_len = min(gpu_comm.MAX_PAYLOAD, remaining)
            chunk = rtl_read_chunk(
                dev,
                args.addr + offset,
                chunk_len,
                verbose=args.verbose_packets,
            )
            out.extend(chunk)
            offset += chunk_len

        print(f"rtl-read returned: addr=0x{args.addr:04x} len={args.length}")
        if args.dump:
            print_hex_dump(bytes(out), args.addr)
        else:
            print(bytes(out).hex(" "))
    finally:
        dev.close()
        print("closed")


def send_roundtrip(args):
    payload = parse_hex_bytes(args.payload)
    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        dev.write_data(args.addr, payload)
        print(f"write_data acked: addr=0x{args.addr:04x} len={len(payload)}")

        data = bytes(dev.read_data(args.addr, len(payload)))
        print("read_data:", data.hex(" "))

        if data != payload:
            print("roundtrip mismatch")
            print("expected:", payload.hex(" "))
            return 1

        print("roundtrip passed")
        return 0
    finally:
        dev.close()
        print("closed")


def send_rtl_roundtrip(args):
    payload = parse_hex_bytes(args.payload)
    read_request = bytes(len(payload))

    dev = open_device(args.port, args.baud, args.timeout_ms)
    try:
        dev.write_data(args.addr, payload)
        print(f"write_data acked: addr=0x{args.addr:04x} len={len(payload)}")

        data = rtl_read_chunk(dev, args.addr, len(read_request), verbose=args.verbose_packets)
        print("read_data:", data.hex(" "))

        if data != payload:
            print("roundtrip mismatch")
            print("expected:", payload.hex(" "))
            return 1

        print("roundtrip passed")
        return 0
    finally:
        dev.close()
        print("closed")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Send Mini-GPU gpu_comm packets to an FPGA serial port."
    )
    parser.add_argument("port", help="serial device, for example /dev/ttyUSB1 or /dev/pts/N")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout-ms", type=int, default=5000)

    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("raw-write", help="send one WRITE_DATA packet without waiting for ACK")
    raw.add_argument("--addr", type=parse_int, default=0x0020)
    raw.add_argument("--payload", default="0f1e2d1f")
    raw.set_defaults(func=send_one_packet)

    write = subparsers.add_parser("write", help="write data and wait for FPGA ACK")
    write.add_argument("--addr", type=parse_int, default=0x0020)
    write.add_argument("--payload", default="0f1e2d1f")
    write.set_defaults(func=send_write)

    split = subparsers.add_parser("split-write", help="write a payload larger than one packet")
    split.add_argument("--addr", type=parse_int, default=0x0000)
    split.add_argument("--extra-bytes", type=int, default=45)
    split.set_defaults(func=send_split_write)

    read = subparsers.add_parser("read", help="read data from the FPGA")
    read.add_argument("--addr", type=parse_int, default=0x0020)
    read.add_argument("--length", type=int, default=4)
    read.set_defaults(func=send_read)

    rtl_read = subparsers.add_parser(
        "rtl-read",
        help="read using the current RTL dummy-payload READ_DATA request, chunking if needed",
    )
    rtl_read.add_argument("--addr", type=parse_int, default=0x0020)
    rtl_read.add_argument("--length", type=int, default=4)
    rtl_read.add_argument("--dump", action="store_true", help="print a 16-byte-wide address dump")
    rtl_read.add_argument("--verbose-packets", action="store_true", help="print each raw request/response packet")
    rtl_read.set_defaults(func=send_rtl_read)

    roundtrip = subparsers.add_parser("roundtrip", help="write data, read it back, and compare")
    roundtrip.add_argument("--addr", type=parse_int, default=0x0020)
    roundtrip.add_argument("--payload", default="0f1e2d1f")
    roundtrip.set_defaults(func=send_roundtrip)

    rtl_roundtrip = subparsers.add_parser(
        "rtl-roundtrip",
        help="write data, read it with the current RTL read request, and compare",
    )
    rtl_roundtrip.add_argument("--addr", type=parse_int, default=0x0020)
    rtl_roundtrip.add_argument("--payload", default="0f1e2d1f")
    rtl_roundtrip.add_argument("--verbose-packets", action="store_true")
    rtl_roundtrip.set_defaults(func=send_rtl_roundtrip)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("port:", args.port)
    try:
        result = args.func(args)
    except Exception as exc:
        print(type(exc).__name__)
        print(exc)
        return 1

    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
