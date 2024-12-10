import os
import subprocess
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    TORCH_LIB_PATH,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

ext_modules = []

ext_modules.append(
    CUDAExtension(
        "fmha",
        sources=[
            "csrc/bindings.cpp",
            "csrc/cpu_fp32.cpp",
        ],
        library_dirs=[TORCH_LIB_PATH],
        runtime_library_dirs=[TORCH_LIB_PATH],
        extra_compile_args=[
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-march=native",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
        ],
    )
)

cmdclass = {}
cmdclass["build_ext"] = BuildExtension.with_options(use_ninja=False)

setup(
    name="fmha_cu",
    version="0.0.1",
    python_requires=">=3.10.0",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
