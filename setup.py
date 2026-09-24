"""
Build the cython extensions of pyprf.

All package metadata is in pyproject.toml. This file only exists because
cython extension modules cannot be declared in pyproject.toml. For a
development installation (compiles the cython code in place):

    pip install -e .[test]
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# The C math library has to be linked explicitly, except on Windows, where it
# is part of the C runtime.
lstLib = [] if sys.platform == 'win32' else ['m']

lstExt = [Extension('pyprf.analysis.' + strNme,
                    sources=['pyprf/analysis/' + strNme + '.pyx'],
                    include_dirs=[np.get_include()],
                    libraries=lstLib,
                    define_macros=[('NPY_NO_DEPRECATED_API',
                                    'NPY_1_7_API_VERSION')])
          for strNme in ['cython_leastsquares',
                         'cython_leastsquares_two',
                         'cython_prf_convolve']]

setup(ext_modules=cythonize(lstExt,
                            compiler_directives={'language_level': 3}))
