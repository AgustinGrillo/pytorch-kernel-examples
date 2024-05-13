# V1.3: TorchScript Custom Operator (Setuptools)

Following: https://pytorch.org/tutorials/advanced/torch_script_custom_ops

## Build
### Install Dependencies
```bash
conda create -n pytorch_v1 python=3.12
conda activate pytorch_v1
conda install -c pytorch pytorch
conda install ninja
```

### Build
```bash
python setup.py build develop
```

## Use
Operator call from Python:

```python
import torch
torch.ops.load_library("custom_mmult.so")
print(torch.ops.custom_ops.custom_mmult)
```

