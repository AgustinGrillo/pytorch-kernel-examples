import torch
import time

torch.ops.load_library("build/libcustom_mmult.so")

N = 200000

custom_ops = torch.ops.custom_ops

a = torch.randn(2, 3)
b = torch.randn(3, 4)


# Default op (torch.mm) with no broadcasting
total_time = 0
for i in range(N):
    start_time = time.time()
    torch.mm(a, b)
    total_time += time.time() - start_time
print("[torch.mm] Average time per matrix multiplication (µs): ", 1e6 * total_time / N)

# Passthrough
total_time = 0
for i in range(N):
    start_time = time.time()
    custom_ops.mmult_passthrough(a, b)
    total_time += time.time() - start_time
print(
    "[custom_ops.mmult_passthrough] Average time per matrix multiplication (µs): ",
    1e6 * total_time / N,
)

# Naive
n = int(N / 10)
total_time = 0
for i in range(n):
    start_time = time.time()
    custom_ops.mmult_naive(a, b)
    total_time += time.time() - start_time
print(
    "[custom_ops.mmult_naive] Average time per matrix multiplication (µs): ",
    1e6 * total_time / n,
)

# MKL
total_time = 0
for i in range(N):
    start_time = time.time()
    custom_ops.mmult_mkl(a, b)
    total_time += time.time() - start_time
print(
    "[custom_ops.mmult_mkl] Average time per matrix multiplication (µs): ",
    1e6 * total_time / N,
)
