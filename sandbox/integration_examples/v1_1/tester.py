import torch

torch.ops.load_library("build/libcustom_mmult.so")

op = torch.ops.custom_ops.custom_mmult

a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = op(a, b)

print(c)
