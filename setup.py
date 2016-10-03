"""Cython setup.

In the terminal, cd to this file's folder then run:
    python setup.py build_ext --inplace
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("py_42_cython_lstsqr.pyx"),
    include_dirs=[np.get_include()]
)
