import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
extensions = [Extension("pyVertexModel.Kg.kg_functions", ["src/pyVertexModel/Kg/kg_functions.pyx"])]

setup(
    name='pyVertexModel',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=cythonize(extensions, **ext_options),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
