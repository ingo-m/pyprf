# -*- coding: utf-8 -*-
"""
Compile cython function for pRF finding.

The same cythonisatin & setup as in setup.py is performed. Necessary for
compilation of cython code before local testing (with pytest).

Call this file with the following bash command:
    python cython_setup.py build_ext --inplace
"""

# Part of py_pRF_mapping library
# Copyright (C) 2017  Omer Faruk Gulban & Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# print('-Compiling cython functions')

# List of external modules (cython):
lstExt = [Extension('pyprf.analysis.cython_leastsquares',
                    sources=['pyprf/analysis/cython_leastsquares.pyx'],
                    include_dirs=[np.get_include()],
                    libraries=['m']
                    ),
          Extension('pyprf.analysis.cython_leastsquares_two',
                    sources=['pyprf/analysis/cython_leastsquares_two.pyx'],
                    include_dirs=[np.get_include()],
                    libraries=['m']
                    ),
          Extension('pyprf.analysis.cython_prf_convolve',
                    sources=['pyprf/analysis/cython_prf_convolve.pyx'],
                    include_dirs=[np.get_include()],
                    libraries=['m']
                    )]

# Compile the cython code for 2D Gaussian convolution of pRF model time
# courses:
setup(ext_modules=cythonize(lstExt))
