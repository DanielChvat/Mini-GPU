import uart_gpu_comm

dev = uart_gpu_comm.Device("/dev/pts/6", 115200)
print("opened:", dev.is_open())

dev.send_raw(b"hello from python\n")

dev.close()
print("opened:", dev.is_open())