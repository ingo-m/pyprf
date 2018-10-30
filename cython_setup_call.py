# -*- coding: utf-8 -*-
"""Cython setup."""

import os
import subprocess as sp


def cython_setup_call():
    """
    Call cython setup.

    Call script to build cython code, using call to bash subprocess.
    """
    # Directory of this file:
    strDir = os.path.dirname(os.path.abspath(__file__))

    # Go down two directories:
    # strDir = os.path.split(strDir)[0]
    # strDir = os.path.split(strDir)[0]

    # Call the script that compiles the cython code:
    sp.call(['python cython_setup.py build_ext --inplace'],
            cwd=strDir,
            shell=True)


if __name__ == '__main__':

    cython_setup_call()
