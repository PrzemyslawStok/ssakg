from setuptools import setup, Extension
from numpy import get_include

extensions = [
    Extension("ssakg_extension", ["ssakg/ssakg_extension.c"],
              include_dirs=[get_include()]),
]

setup(ext_modules=extensions)