from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "src/pyVertexModel/Kg/kg_functions.pyx",  # Path to your .pyx file
        language_level="3"  # Ensure Python 3 compatibility
    )
)