import torch
import vgpu
import torch.utils.benchmark as benchmark
from torch.profiler import profile, schedule, record_function, ProfilerActivity
import time
import numpy as np

N = 10

m = int(3000)
k = int(3000)
n = int(3000)


# Create two random matrices in float32
a = torch.randn(m, k)
b = torch.randn(k, n)

operators = [
    torch.mm,
    # torch.vgpu.ops.mmult_passthrough,
    # torch.vgpu.ops.mmult_naive,
    # torch.vgpu.ops.mmult_naive_multithreaded,
    torch.vgpu.ops.mmult_mkl,
]

# torch.vgpu.utils.set_num_threads(4)

# Benchmark
for operator in operators:
    print(f"[{operator.__module__}.{operator.__name__}] Warming up...")
    for i in range(N):
        operator(a, b)  # Warm up

    start_time = None
    times = np.array([])
    for i in range(N):
        start_time = time.perf_counter()
        operator(a, b)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        times = np.append(times, total_time)

    print("Raw results:")
    print(
        f"[{operator.__module__}.{operator.__name__}] Average time per matrix multiplication (µs): { 1e6 * np.mean(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Median time per matrix multiplication (µs): { 1e6 * np.median(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Standard deviation of time per matrix multiplication (µs): { 1e6 * np.std(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Min / Max time per matrix multiplication (µs): { 1e6 * np.min(times):.2f} / { 1e6 * np.max(times):.2f}"
    )
    print("Results without outliers:")
    times = times[times < np.mean(times) + 2 * np.std(times)]
    print(
        f"[{operator.__module__}.{operator.__name__}] Average time per matrix multiplication (µs): { 1e6 * np.mean(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Median time per matrix multiplication (µs): { 1e6 * np.median(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Standard deviation of time per matrix multiplication (µs): { 1e6 * np.std(times):.2f}"
    )
    print(
        f"[{operator.__module__}.{operator.__name__}] Min / Max time per matrix multiplication (µs): { 1e6 * np.min(times):.2f} / { 1e6 * np.max(times):.2f}"
    )


# for operator in operators:
#     for i in range(N):
#         operator(a, b)  # Warm up
#     with profile(
#         activities=[ProfilerActivity.CPU],
#     ) as prof:
#         for i in range(N):
#             with record_function("operator_call"):
#                 operator(a, b)
#
#     print(prof.key_averages())
