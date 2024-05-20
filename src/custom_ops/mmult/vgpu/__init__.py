import torch
import sys

current_module = sys.modules[__name__]

# Load the custom library
torch.ops.load_library("build/libcustom_mmult.so")
custom_ops = torch.ops.custom_ops

# Define custom device name
torch.utils.rename_privateuse1_backend("vgpu")
torch._register_device_module("vgpu", current_module)

# TODO: workaround (?)
import vgpu.utils
import vgpu.ops
