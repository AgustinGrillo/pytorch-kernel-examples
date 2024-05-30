import torch
import vgpu

m = 10
k = 10
n = 10

# Create two random matrices in float32
a_cpu = torch.randn(m, k)
b_cpu = torch.randn(k, n)

# Custom op
c_cpu = torch.vgpu.ops.mmult_mkl(a_cpu, b_cpu)
print(c_cpu)

custom_device = "vgpu:0"
a_vgpu = torch.rand(m, k, device=custom_device)
b_vgpu = torch.rand(k, n, device=custom_device)
c_vgpu = torch.mm(a_vgpu, b_vgpu)
print(c_vgpu.cpu())
