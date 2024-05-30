import torch

vgpu = torch.ops.vgpu


def mmult_mkl(a, b):
    return vgpu.mmult_mkl(a, b)


def mmult_naive(a, b):
    return vgpu.mmult_naive(a, b)


def mmult_naive_multithreaded(a, b):
    return vgpu.mmult_naive_multithreaded(a, b)
