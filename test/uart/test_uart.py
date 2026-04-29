import sys
import gpu_comm


def make_payload(n):
    return bytes(i & 0xFF for i in range(n))


def send_one_packet(port):
    packet = gpu_comm.build_packet(
        gpu_comm.Command.WRITE_DATA,
        0x0020,
        bytes([0x0F, 0x1E, 0x2D, 0x1F]),
    )

    print("packet:", packet.hex(" "))
    print("packet length:", len(packet))

    dev = gpu_comm.Device(port, 115200, 5000)
    print("opened:", dev.is_open())

    dev.send_raw(packet)
    print("sent one hard-coded packet")

    dev.close()
    print("closed")


def send_split_write(port):
    max_payload = gpu_comm.MAX_PAYLOAD
    payload = make_payload(max_payload + 10)

    print("max payload:", max_payload)
    print("payload length:", len(payload))
    print("expected packets: 2")
    print("expected first packet bytes:", 7 + max_payload)
    print("expected second packet bytes:", 7 + 3)

    dev = gpu_comm.Device(port, 115200, 5000)
    print("opened:", dev.is_open())

    dev.write_data(0x0000, payload)
    print("sent split write_data payload")

    dev.close()
    print("closed")


def main():
    if len(sys.argv) not in (2, 3):
        print(f"usage: python3 {sys.argv[0]} /dev/pts/N [--split]")
        return 1

    port = sys.argv[1]

    print("port:", port)

    if len(sys.argv) == 3:
        if sys.argv[2] != "--split":
            print(f"unknown option: {sys.argv[2]}")
            return 1

        send_split_write(port)
    else:
        send_one_packet(port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
