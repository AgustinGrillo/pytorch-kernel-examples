import torch
import time

torch.ops.load_library("build/libcustom_mmult.so")

N = 10000

m = 8
k = 16 * 128
n = 128

custom_ops = torch.ops.custom_ops

# Create two random matrices in float32
a = torch.randn(m, k)
b = torch.randn(k, n)

operators = [
    torch.mm,
    custom_ops.mmult_passthrough,
    # custom_ops.mmult_naive,
    custom_ops.mmult_mkl,
]


# Warm up
print("Warming up...")
for i in range(N):
    start_time = time.time()
    torch.mm(a, b)

# Benchmark
for operator in operators:
    start_time = time.time()
    for i in range(N):
        operator(a, b)
    total_time = time.time() - start_time
    print(
        f"[{operator.__module__}.{operator.__name__}] Average time per matrix multiplication (µs): { 1e6 * total_time / N }"
    )
