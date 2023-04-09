from setuptools import setup, Extension

setup(
    name="aptedModule",
    version="0.1",
    ext_modules=[Extension("aptedModule", ["aptedModule.cpp"])],
)
