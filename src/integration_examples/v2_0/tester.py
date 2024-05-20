import torch
import vgpu


# custom_device = torch.device("vgpu")
custom_device = "vgpu:0"

# a = torch.randn(2, 3).to(custom_device)
# b = torch.randn(3, 4).to(custom_device)
a = torch.randn(2, 3, device=custom_device)
b = torch.randn(3, 4, device=custom_device)
c = torch.mm(a, b)

print(c)
