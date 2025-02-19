from setuptools import setup, Extension, find_packages

VERSION = '0.2.0'
DESCRIPTION = 'Sequential Structural Associative Knowledge Graph (ssakg)'
LONG_DESCRIPTION = open('README_PYPI.md').read()

def get_numpy_include():
    try:
        from numpy import get_include
        return [get_include()]
    except ImportError:
        return []

extensions = [
    Extension("ssakg_extension", ["ssakg/ssakg_extension.c"],
              include_dirs=get_numpy_include()),
]

setup(
    name='ssakg',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=extensions,
    include_package_data=True,
    install_requires=['numpy~=2.2.2', "networkx~=3.2.1", "matplotlib~=3.8.2", "pandas~=2.2.0", "tabulate~=0.9.0",
                      "PyPrind~=2.11.3", "seaborn~=0.13.2"],
    url='https://github.com/PrzemyslawStok/ssakg',
    license='Apache 2.0',
    author='Przemysław Stokłosa',
    author_email='przemyslaw.stoklosa@gmail.com',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords=["semantic memory", "structural graphs", "graph density", "sequence storage", "sequence retrieval",
              "context based memory", "graph based memory", "mirna sequences"],
    python_requires='>=3.10,<=3.12.7',
)
