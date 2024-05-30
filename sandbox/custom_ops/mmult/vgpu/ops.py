import torch

custom_ops = torch.ops.custom_ops


def mmult_passthrough(a, b):
    return custom_ops.mmult_passthrough(a, b)


def mmult_mkl(a, b):
    return custom_ops.mmult_mkl(a, b)


def mmult_naive(a, b):
    return custom_ops.mmult_naive(a, b)


def mmult_naive_multithreaded(a, b):
    return custom_ops.mmult_naive_multithreaded(a, b)
