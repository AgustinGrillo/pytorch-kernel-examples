import torch
import vgpu
from torch.profiler import profile, schedule, record_function, ProfilerActivity
import time
import numpy as np


torch.set_num_threads(1)
print(f"Number of threads: {torch.get_num_threads()}")

N = 100

m = int(8)
k = int(2**15)  # 8 a 32k
n = int(128)

# custom_device = torch.device("vgpu")
custom_device = "vgpu:0"

a = torch.rand(m, k, device=custom_device)
b = torch.rand(k, n, device=custom_device)

# print(f"Warming up...")
for i in range(N):
    torch.mm(a, b)  # Warm up

start_time = time.perf_counter()
for i in range(N):
    torch.mm(a, b)
end_time = time.perf_counter()
total_time = end_time - start_time
mean_time = total_time / N
print(f"Average time per matrix multiplication (µs): { 1e6 * mean_time:.2f}")

# for i in range(N):
#     torch.mm(a, b)  # Warm up
# with profile(
#     activities=[ProfilerActivity.CPU],
# ) as prof:
#     for i in range(N):
#         with record_function("operator_call"):
#             operator(a, b)
#
# print(prof.key_averages())
