from setuptools import setup, Extension
import numpy as np

setup(
    name="wgwl",
    version="0.1",
    ext_modules=[
        Extension(
            "wgwl",
            ["module.cpp", "lib/hungarian/hungarian.cpp"],
            extra_compile_args=["/std:c++17", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    include_dirs=[np.get_include(), "./lib/"],
)
