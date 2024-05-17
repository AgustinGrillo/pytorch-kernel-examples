import torch

torch.ops.load_library("build/libcustom_mmult.so")

# op = torch.ops.custom_ops.mm

# custom_device = torch.device("privateuseone")
custom_device = "privateuseone:0"

# a = torch.randn(2, 3).to(custom_device)
# b = torch.randn(3, 4).to(custom_device)
a = torch.randn(2, 3, device=custom_device)
b = torch.randn(3, 4, device=custom_device)
c = torch.mm(a, b)

print(c)
