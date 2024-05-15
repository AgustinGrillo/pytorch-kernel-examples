import torch
import time

torch.ops.load_library("build/libcustom_mmult.so")

N = 200000

custom_ops = torch.ops.custom_ops

# Create two random matrices in float32
a = torch.randn(2, 3)
b = torch.randn(3, 4)

operators = [
    torch.mm,
    custom_ops.mmult_passthrough,
    custom_ops.mmult_naive,
    custom_ops.mmult_mkl,
]


for operator in operators:
    total_time = 0
    for i in range(N):
        start_time = time.time()
        operator(a, b)
        total_time += time.time() - start_time
    print(
        f"[{operator.__module__}.{operator.__name__}] Average time per matrix multiplication (µs): { 1e6 * total_time / N }"
    )
