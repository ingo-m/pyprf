"""
pyprf setup.

For development installation:
    pip install -e /path/to/pRF_mapping
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class clssBldNmp(build_ext):
    """
    Class for handling numpy dependency for cython setup.

    Subclass of `setuptools.command.build_py.build_py`

    In order to install the cython extensions, the
    `ext_modules=[Extension(...)]` argument of the `setuptools.setup(...)`
    function is needed. Problematically, the expression `numpy.get_include()`
    needs to be passed into this function. Thus, there is a `numpy` dependency
    in this `setup.py` file. Thus, `import numpy` needs to be executed before
    reaching the `setuptools.setup(...)` command, which specifies the numpy
    dependency - a chicken-and-egg problem. With this class, we add a custom
    build command to the `setuptools.setup(...)` command. This is somewhat of a
    workaround, but it seems necessary in order to make pyprf installable
    through pip with full cython functionality.

    This solution is based on the following references:
    https://stackoverflow.com/a/21621689
    https://seasonofcode.com/posts/how-to-add-custom-build-steps-and-commands-to-setuppy.html
    """

    def finalize_options(self):
        """Workaround for numpy dependency of cython setup."""
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


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
      install_requires=['numpy', 'scipy', 'nibabel', 'pillow', 'cython',
                        'tensorflow'],
      setup_requires=['numpy'],
      keywords=['pRF', 'fMRI', 'retinotopy'],
      long_description=long_description,
      entry_points={
          'console_scripts': [
              'pyprf = pyprf.analysis.__main__:main',
              ]},
      cmdclass={'build_ext': clssBldNmp},
      ext_modules=[Extension('pyprf.analysis.cython_leastsquares',
                             ['pyprf/analysis/cython_leastsquares.pyx'],
                             include_dirs=[numpy.get_include()]
                             )],
      )

# Load module to setup python:
# from cython_leastsquares_setup_call import setup_cython  #noqa

# Compile cython code:
# setup_cython()
