import torch
import vgpu
from torch.profiler import profile, schedule, record_function, ProfilerActivity
import time
import numpy as np


N = 10

m = int(10)
k = int(10)
n = int(10)

# custom_device = torch.device("vgpu")
custom_device = "vgpu:0"

a = torch.rand(m, k, device=custom_device)
b = torch.rand(k, n, device=custom_device)

print(f"Warming up...")
for i in range(N):
    torch.mm(a, b)  # Warm up

start_time = None
times = np.array([])
for i in range(N):
    start_time = time.perf_counter()
    torch.mm(a, b)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    times = np.append(times, total_time)

print("Raw results:")
print(f"Average time per matrix multiplication (µs): { 1e6 * np.mean(times):.2f}")
print(f"Median time per matrix multiplication (µs): { 1e6 * np.median(times):.2f}")
print(
    f"Standard deviation of time per matrix multiplication (µs): { 1e6 * np.std(times):.2f}"
)
print(
    f"Min / Max time per matrix multiplication (µs): { 1e6 * np.min(times):.2f} / { 1e6 * np.max(times):.2f}"
)
print("Results without outliers:")
times = times[times < np.mean(times) + 2 * np.std(times)]
print(f"Average time per matrix multiplication (µs): { 1e6 * np.mean(times):.2f}")
print(f"Median time per matrix multiplication (µs): { 1e6 * np.median(times):.2f}")
print(
    f"Standard deviation of time per matrix multiplication (µs): { 1e6 * np.std(times):.2f}"
)
print(
    f"Min / Max time per matrix multiplication (µs): { 1e6 * np.min(times):.2f} / { 1e6 * np.max(times):.2f}"
)

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
