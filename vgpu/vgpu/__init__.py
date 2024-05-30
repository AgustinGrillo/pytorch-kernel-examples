import torch
import sys
import os

current_module = sys.modules[__name__]
file_path = os.path.dirname(os.path.abspath(__file__))

# Load the custom library
torch.ops.load_library(file_path + "/../build/libvgpu.so")
custom_ops = torch.ops.custom_ops

# Define custom device name
torch.utils.rename_privateuse1_backend("vgpu")
unsupported_dtype = []
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True
)
torch._register_device_module("vgpu", current_module)

# TODO: workaround (?)
import vgpu.utils
import vgpu.ops
