# V2.0: Adding a new PyTorch backend

Following PyTorch tutorials: 
- [Extending dispatcher for a new backend in C++](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)
- [Facilitating New Backend Integration by PrivateUse1](https://pytorch.org/tutorials/advanced/privateuseone.html)

## Build
### Install Dependencies
```bash
conda env create -n pytorch_extension -f environment.yml
conda activate pytorch_extension
```

### Build
```bash
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"  .. 
make -j
```

## Run
```bash
python tester.py
```

