# V1.2: TorchScript Custom Operator (JIT compilation)

Following PyTorch tutorial: [Extending TorchScript with Custom C++ Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops)

## Build
### Install Dependencies
```bash
conda create -n pytorch_v1 python=3.12
conda activate pytorch_v1
conda install -c pytorch pytorch
conda install ninja
```

## Use
Build and call from Python:

```python
import torch.utils.cpp_extension
torch.utils.cpp_extension.load(
    name="custom_mmult",
    sources=["custom_mmult_op.cpp"],
    is_python_module=False,
    verbose=True,
)
print(torch.ops.custom_ops.custom_mmult)
```
