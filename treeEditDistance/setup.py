from setuptools import setup, Extension
import numpy as np

import platform

# Determine the platform we're running on
system = platform.system()
if system == "Windows":
    # Windows-specific compile/link flags
    extra_compile_args = ["/std:c++17", "/openmp"]
    extra_link_args = ["/openmp"]
else:
    # Linux/Unix-specific compile/link flags
    extra_compile_args = ["-std=c++17", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

setup(
    name="wgwl",
    version="0.1",
    ext_modules=[
        Extension(
            "wgwl",
            ["module.cpp", "lib/hungarian/hungarian.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    include_dirs=[np.get_include(), "./lib/"],
)
