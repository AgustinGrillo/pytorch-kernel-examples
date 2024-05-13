import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="custom_mmult",
    sources=["custom_mmult_op.cpp"],
    is_python_module=False,
    verbose=True,
)

op = torch.ops.custom_ops.custom_mmult

a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = op(a, b)

print(c)
