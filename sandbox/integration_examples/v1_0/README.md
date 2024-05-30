# V1.0: Custom C++ extension (Setuptools)

Following PyTorch tutorial: [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions)

> Note: This is an old version. \
> v1_3 is a more up-to-date alternative. \
> Uses `TORCH_LIBRARY` to bind operators (instead of using PyBind)  \
> `TORCH_LIBRARY` makes the oprator available for use in both eager Python as well as in TorchScript, \
> while `PYBIND11_MODULE` lets you bind C++ to Python only.

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
python setup.py install
```

or

```bash
python setup.py build develop
```

## Use
Operator call from Python:

```python
import torch
import custom_mmult
print(custom_mmult.custom_mmult)
```

