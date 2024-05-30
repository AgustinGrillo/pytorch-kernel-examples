import torch

# Load the custom library
torch.ops.load_library("build/libcustom_backend.so")

# Define custom device name
torch.utils.rename_privateuse1_backend("vgpu")
unsupported_dtype = []
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True
)
