"""
pyprf setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

import numpy as np
from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

with open('README.rst') as f:
    long_description = f.read()

# Whereas install_requires metadata is automatically analyzed by pip during an
# install (i.e. also when installing from pypi), requirements files are not,
# and only are used when a user specifically installs them using pip install
# -r. Therefore, we pin versions here.

# From the cython documentation:
# "Note also that if you use setuptools instead of distutils, the default
# action when running python setup.py install is to create a zipped egg file
# which will not work with cimport for pxd files when you try to use them from
# a dependent package. To prevent this, include zip_safe=False in the arguments
# to setup()."
# Source:
# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html

setup(name='pyprf',
      version='1.3.13',
      description=('A free & open source python tool for population receptive \
                    field analysis of fMRI data.'),
      url='https://github.com/ingo-m/pyprf',
      download_url='https://github.com/ingo-m/pyprf/archive/v1.3.13.tar.gz',
      author='Ingo Marquardt',
      author_email='ingo.marquardt@gmx.de',
      license='GNU General Public License Version 3',
      install_requires=['numpy==1.15.1', 'scipy==1.1.0', 'nibabel==2.2.1',
                        'pillow==5.0.0', 'cython==0.27.1',
                        'tensorflow==1.4.0'],
      # setup_requires=['numpy'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      packages=['pyprf.analysis'],
      py_modules=['pyprf.analysis'],
      entry_points={
          'console_scripts': [
              'pyprf = pyprf.analysis.__main__:main',
              ]},
      ext_modules=cythonize(
          [Extension('pyprf.analysis.cython_leastsquares',  # ['pyprf/analysis/cython_leastsquares.c'],
                     sources=['pyprf/analysis/cython_leastsquares.pyx'],
                     include_dirs=[np.get_include()],
                     libraries=['m']
                     ),
           Extension('pyprf.analysis.cython_leastsquares_two',  # ['pyprf/analysis/cython_leastsquares_two.c'],
                     sources=['pyprf/analysis/cython_leastsquares_two.pyx'],
                     include_dirs=[np.get_include()],
                     libraries=['m']
                     ),
           Extension('pyprf.analysis.cython_prf_convolve',  # ['pyprf/analysis/cython_prf_convolve.c'],
                     sources=['pyprf/analysis/cython_prf_convolve.pyx'],
                     include_dirs=[np.get_include()],
                     libraries=['m']
                     )]
          ),
      )

# Load module to setup python:
# from pyprf.analysis.cython_leastsquares_setup_call import setup_cython_lstsq
# from pyprf.analysis.cython_prf_convolve_setup_call import setup_cython_conv

# Compile cython code:
# setup_cython_lstsq()
# setup_cython_conv()
