"""
pyprf setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

import os
import subprocess
from setuptools import setup


with open('README.md') as f:
    long_description = f.read()

setup(name='pyprf',
      version='1.1.0',
      description=('A free & open source python tool for population receptive \
                    field analysis of fMRI data.'),
      url='https://github.com/ingo-m/pyprf',
      author='Ingo Marquardt',
      license='GNU General Public License Version 3',
      packages=['pyprf.analysis'],
      install_requires=['numpy', 'scipy', 'nibabel', 'pillow', 'cython'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      entry_points={
          'console_scripts': [
              'pyprf = pyprf.analysis.__main__:main',
              ]},
      )

# Cython setup

# Directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))

# Path of cython setup script:
strDirCy = (strDir + '/pyprf/analysis')

# Call the script that compiles the cython code:
subprocess.call([('python cython_leastsquares_compile.py build_ext '
                  + '--inplace')],
                cwd=strDirCy,
                shell=True)

# print('Cython code could not be compiled. You can still use numpy version.')
