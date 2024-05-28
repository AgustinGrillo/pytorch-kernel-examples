import torch
import vgpu
import torch.utils.benchmark as benchmark
from torch.profiler import profile, schedule, record_function, ProfilerActivity
import time
import numpy as np

N = 100

m = int(8)
k = int(2**15)  # 8 a 32k
n = int(128)

torch.set_num_threads(1)
print(f"Number of threads: {torch.get_num_threads()}")

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

    start_time = time.perf_counter()
    for i in range(N):
        operator(a, b)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    mean_time = total_time / N
    print(
        f"[{operator.__module__}.{operator.__name__}] Average time per matrix multiplication (µs): { 1e6 * mean_time:.2f}"
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
