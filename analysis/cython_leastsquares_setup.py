# -*- coding: utf-8 -*-
"""
Compile cython function for pRF finding.

Call this file with the following bash command:
    python cython_leastsquares_setup.py build_ext --inplace
"""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Omer Faruk Gulban & Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

print('-Compiling cython function')

# Compile the code:
setup(ext_modules=cythonize('cython_leastsquares.pyx'),
      include_dirs=[np.get_include()])
