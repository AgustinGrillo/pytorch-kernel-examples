import torch
import timeit

torch.ops.load_library("build/libcustom_mmult.so")

m = 1000
k = 1000
n = 1000

custom_ops = torch.ops.custom_ops

# Create two random matrices in float32
a = torch.randn(m, k)
b = torch.randn(k, n)

# Multithreading setup and execution
custom_ops.set_num_threads(1)
print(timeit.timeit(lambda: custom_ops.mmult_naive_multithreaded(a, b), number=2))

custom_ops.set_num_threads(2)
print(timeit.timeit(lambda: custom_ops.mmult_naive_multithreaded(a, b), number=2))
