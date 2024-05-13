# V1.4: TorchScript Custom Operator (Setuptools)

Following PyTorch tutorial: [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions)

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

