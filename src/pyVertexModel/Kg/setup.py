from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [Extension("kg_functions", ["kg_functions.pyx"])]

setup(
    ext_modules=cythonize(extensions, gdb_debug=True),
    include_dirs=[np.get_include()],
)