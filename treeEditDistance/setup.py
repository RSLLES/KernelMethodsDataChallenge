from setuptools import setup, Extension
import numpy as np

setup(
    name="treeEditDistance",
    version="0.1",
    ext_modules=[
        Extension(
            "treeEditDistance", ["treeEditDistance.cpp", "lib/hungarian/hungarian.cpp"]
        )
    ],
    include_dirs=[np.get_include()],
)
