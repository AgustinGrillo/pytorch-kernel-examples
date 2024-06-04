# Custom PyTorch extension

## Dependencies
```bash
conda env create -n pytorch_extension -f environment.yml
conda activate pytorch_extension
```

Intel MKL:
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/l_onemkl_p_2024.1.0.695.sh
sudo sh ./l_onemkl_p_2024.1.0.695.sh
```

## Build
```bash
mkdir vgpu/build && cd vgpu/build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"  ../csrc 
make -j
```

## Run
```bash
python vgpu/example.py
```

