"""
pRF mapping setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

# try:
from setuptools import setup
# except ImportError:
#     from distutils.core import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='py_pRF_mapping',
      version='1.1.0',
      description=('A free & open source python tool for population receptive \
                    field analysis of fMRI data.'),
      url='https://github.com/ingo-m/py_pRF_mapping',
      author='Ingo Marquardt',
      license='GNU General Public License Version 3',
      packages=['py_pRF_mapping.analysis'],
      install_requires=['numpy', 'scipy', 'nibabel'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      entry_points={
          'console_scripts': [
              'py_pRF_mapping = py_pRF_mapping.analysis.__main__:main',
              ]},
      )
