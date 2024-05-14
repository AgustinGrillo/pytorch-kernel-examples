import torch

torch.ops.load_library("custom_mmult.so")

op = torch.ops.custom_ops.custom_mmult

a_dense = torch.randn(2, 3)
b_dense = torch.randn(3, 4)

a_sparse = torch.randn(2, 3).to_sparse()

# a_mkl = torch.randn(2, 3).to_mkldnn()

print(op(a_dense, b_dense))

print(op(a_sparse, b_dense))

# print(op(a_mkl, b_dense))
