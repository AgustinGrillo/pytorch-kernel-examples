import torch

custom_ops = torch.ops.custom_ops


def set_num_threads(num_threads):
    return custom_ops.set_num_threads(num_threads)
