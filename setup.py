import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
extensions = [Extension("kg_functions", ["src/pyVertexModel/Kg/kg_functions.pyx"])]

setup(
    name='pyVertexModel',
    version='0.0.1a0',
    packages=['src', 'src.pyVertexModel', 'src.pyVertexModel.Kg', 'pyVertexModel'],
    url='',
    license='',
    author='Pablo Vicente Munuera',
    author_email='p.munuera@ucl.ac.uk',
    description='',
    ext_modules=cythonize(extensions, **ext_options),
    include_dirs=[np.get_include()],
)
