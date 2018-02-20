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

# Whereas install_requires metadata is automatically analyzed by pip during an
# install (i.e. also when installing from pypi), requirements files are not,
# and only are used when a user specifically installs them using pip install
# -r. Therefore, we pin versions here.

setup(name='pyprf',
      version='1.3.5',
      description=('A free & open source python tool for population receptive \
                    field analysis of fMRI data.'),
      url='https://github.com/ingo-m/pyprf',
      download_url='https://github.com/ingo-m/pyprf/archive/v1.3.5.tar.gz',
      author='Ingo Marquardt',
      author_email='ingo.marquardt@gmx.de',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel', 'pillow==5.0.0',
                        'cython==0.27.1', 'tensorflow==1.4.0'],
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
                             ['pyprf/analysis/cython_leastsquares.c'],
                             include_dirs=[np.get_include()]
                             )],
      )

# Load module to setup python:
# from cython_leastsquares_setup_call import setup_cython  #noqa

# Compile cython code:
# setup_cython()
