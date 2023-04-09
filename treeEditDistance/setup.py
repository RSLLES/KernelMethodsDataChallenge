from setuptools import setup, Extension

setup(
    name="treeEditDistance",
    version="0.1",
    ext_modules=[
        Extension(
            "treeEditDistance", ["treeEditDistance.cpp", "lib/hungarian/hungarian.cpp"]
        )
    ],
)
