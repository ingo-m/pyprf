"""
pyprf setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

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


from cython_leastsquares_setup_call import setup_cython

setup_cython()
