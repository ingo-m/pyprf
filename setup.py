"""
pyprf setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

import numpy as np
from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext

with open('README.md') as f:
    long_description = f.read()

setup(name='pyprf',
      version='1.3.0',
      description=('A free & open source python tool for population receptive \
                    field analysis of fMRI data.'),
      url='https://github.com/ingo-m/pyprf',
      download_url='https://github.com/ingo-m/pyprf/archive/v1.3.0.tar.gz',
      author='Ingo Marquardt',
      author_email='ingo.marquardt@gmx.de',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel', 'pillow', 'cython',
                        'tensorflow'],
      # setup_requires=['numpy'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      packages=['pyprf.analysis'],
      py_modules=['pyprf.analysis'],
      entry_points={
          'console_scripts': [
              'pyprf = pyprf.analysis.__main__:main',
              ]},
      ext_modules=[Extension('pyprf.analysis.cython_leastsquares',
                             ['pyprf/analysis/cython_leastsquares.pyx'],
                             include_dirs=[np.get_include()]
                             )],
      )

# Load module to setup python:
# from cython_leastsquares_setup_call import setup_cython  #noqa

# Compile cython code:
# setup_cython()
