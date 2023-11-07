import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension("kg_functions", ["kg_functions.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)