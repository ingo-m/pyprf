# -*- coding: utf-8 -*-
"""Cython setup."""

import os
import subprocess

# Directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))

# Call the script that compiles the cython code:
subprocess.call(['python cython_leastsquares_setup.py build_ext --inplace'],
                cwd=strDir,
                shell=True)
