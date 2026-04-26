import uart_gpu_comm

dev = uart_gpu_comm.Device("/dev/pts/6", 115200, 5000)

try:
    data = dev.recv_raw(12)
    print("received:", data)
finally:
    dev.close()