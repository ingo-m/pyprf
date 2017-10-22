# -*- coding: utf-8 -*-
"""Cython setup."""

import os
import subprocess as sp


def setup_cython():
    """
    Cython setup.

    Call script to build cython code, using call to bash subprocess.
    """
    # Directory of this file:
    strDir = os.path.dirname(os.path.abspath(__file__))

    # Call the script that compiles the cython code:
    sp.call(['python cython_leastsquares_setup.py build_ext --inplace'],
            cwd=strDir,
            shell=True)
