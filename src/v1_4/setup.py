from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="custom_mmult",
    ext_modules=[
        CppExtension(
            "custom_mmult",
            ["custom_mmult_op.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
