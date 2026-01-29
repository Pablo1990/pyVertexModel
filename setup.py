import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
extensions = [Extension("src.pyVertexModel.Kg.kg_functions", ["src/pyVertexModel/Kg/kg_functions.pyx"])]

setup(
    name='pyVertexModel',
    packages=['src', 'src.pyVertexModel', 'src.pyVertexModel.Kg', 'pyVertexModel'],
    ext_modules=cythonize(extensions, **ext_options),
    include_dirs=[np.get_include()],
)
