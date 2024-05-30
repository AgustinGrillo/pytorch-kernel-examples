import torch
import custom_mmult

op = custom_mmult.custom_mmult

a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = op(a, b)

print(c)
