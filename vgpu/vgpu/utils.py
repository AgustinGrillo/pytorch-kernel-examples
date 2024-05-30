import torch

vgpu = torch.ops.vgpu


def set_num_threads(num_threads):
    return vgpu.set_num_threads(num_threads)
