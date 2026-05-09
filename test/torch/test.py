import torch
import torch_mini_gpu

torch_mini_gpu.connect("/dev/ttyUSB1", memory_size=65536)

x = torch.arange(256, dtype=torch.int8)
y = x.to("minigpu")

print(y.to("cpu"))
print(y.view(16, 16).to("cpu"))

torch_mini_gpu.disconnect()
