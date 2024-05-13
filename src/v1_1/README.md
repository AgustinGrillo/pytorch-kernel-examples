# V1.1: TorchScript Custom Operator

Following: https://pytorch.org/tutorials/advanced/torch_script_custom_ops

## Build
### Install PyTorch
```bash
conda create -n pytorch_v1 python=3.12
conda activate pytorch_v1
conda install -c pytorch pytorch
```

### Build
```bash
mkdir build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"  ..
make -j
```

## Use
Function call from Python:

```python
import torch
torch.ops.load_library("build/libcustom_mmult.so")
print(torch.ops.custom_ops.custom_mmult)
```
