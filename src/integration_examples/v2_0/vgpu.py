import torch

# Load the custom library
torch.ops.load_library("build/libcustom_backend.so")

# Define custom device name
torch.utils.rename_privateuse1_backend("vgpu")
