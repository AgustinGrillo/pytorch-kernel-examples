import torch
import vgpu
import timeit

m = 1000
k = 1000
n = 1000

# Create two random matrices in float32
a = torch.randn(m, k)
b = torch.randn(k, n)

# Multithreading setup and execution
torch.vgpu.utils.set_num_threads(1)
print(timeit.timeit(lambda: torch.vgpu.ops.mmult_naive_multithreaded(a, b), number=2))

torch.vgpu.utils.set_num_threads(2)
print(timeit.timeit(lambda: torch.vgpu.ops.mmult_naive_multithreaded(a, b), number=2))
