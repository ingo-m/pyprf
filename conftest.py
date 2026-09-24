"""Shared pytest fixtures for the tests in `pyprf/analysis`."""

import os
import pathlib
import shutil

import numpy as np
import pytest

from pyprf.analysis import pyprf_main
from pyprf.analysis import utilities as util


# Decimal places to round to before comparing template and test results:
varRnd = 3


class TestRunner:
    """
    Run the pyprf pipeline with the config files of a testing directory.

    The paths in the testing config files are relative to `pyprf/analysis`
    (e.g. '/testing/exmpl_data_mask.nii.gz'). The runner writes a copy of the
    config file with absolute paths to a temporary directory, so that all
    output (results, model time courses, hdf5 files) is written to the
    temporary directory, and not into the package directory.

    Some config files create the pRF model time courses, others load the
    model time courses created by a previous run. `dicDep` maps the version
    abbreviation of a config (e.g. 'cy') to the version that has to run
    first (e.g. 'np'), so that each test can also be run on its own.
    """

    __test__ = False

    def __init__(self, strDirTest, strDirTmp, dicDep):
        # Paths are written into the config files with forward slashes, because
        # config values are parsed as python literals, where backslashes (in
        # Windows paths) would be interpreted as escape sequences.
        self.strDirTest = pathlib.Path(strDirTest).as_posix()
        self.strDirTmp = pathlib.Path(strDirTmp).as_posix()
        self.dicDep = dicDep
        self.setDone = set()

        # Name of the testing directory, as used in the config files (e.g.
        # 'testing' or 'testing_two_predictors'):
        self.strNme = os.path.basename(strDirTest)

        os.makedirs(os.path.join(strDirTmp, 'result'), exist_ok=True)

        # In hdf5 mode, the functional data are converted to hdf5 files next
        # to the input nii files. Therefore, the functional data are copied to
        # the temporary directory.
        for strTmp in os.listdir(strDirTest):
            if strTmp.startswith('exmpl_data_func_'):
                shutil.copy(os.path.join(strDirTest, strTmp), strDirTmp)

    def run(self, strVrsn):
        """Run pyprf with config file `config_testing_<strVrsn>.csv`."""
        if strVrsn in self.setDone:
            return

        if strVrsn in self.dicDep:
            self.run(self.dicDep[strVrsn])

        strCsvIn = os.path.join(self.strDirTest,
                                'config_testing_{}.csv'.format(strVrsn))
        with open(strCsvIn, 'r') as objFle:
            strCnfg = objFle.read()

        # Output and functional data go to the temporary directory, all other
        # input files (mask, stimulus PNGs) are read from the testing
        # directory:
        strPre = "'/" + self.strNme + '/'
        strCnfg = strCnfg.replace(strPre + 'result/',
                                  "'" + self.strDirTmp + '/result/')
        strCnfg = strCnfg.replace(strPre + 'exmpl_data_func_',
                                  "'" + self.strDirTmp + '/exmpl_data_func_')
        strCnfg = strCnfg.replace(strPre, "'" + self.strDirTest + '/')

        strCsvOut = os.path.join(self.strDirTmp,
                                 'config_testing_{}.csv'.format(strVrsn))
        with open(strCsvOut, 'w') as objFle:
            objFle.write(strCnfg)

        pyprf_main.pyprf(strCsvOut)

        self.setDone.add(strVrsn)

    def assert_results(self, strVrsn, lstMaps):
        """Compare results of a run with the template results."""
        for strMap in lstMaps:

            aryTmpl, _, _ = util.load_nii(os.path.join(
                self.strDirTest, 'exmpl_data_results_{}.nii.gz'.format(strMap)))

            aryTest, _, _ = util.load_nii(os.path.join(
                self.strDirTmp, 'result',
                'pRF_test_results_{}_{}.nii.gz'.format(strVrsn, strMap)))

            aryTmpl = np.around(aryTmpl, decimals=varRnd).astype(np.float32)
            aryTest = np.around(aryTest, decimals=varRnd).astype(np.float32)

            np.testing.assert_array_equal(
                aryTest, aryTmpl,
                err_msg='Version {}, map {}'.format(strVrsn, strMap))


@pytest.fixture(scope='module')
def test_runner(request, tmp_path_factory):
    """
    Test runner for the testing directory of the requesting test module.

    The test module has to define `dicDep` (see `TestRunner`).
    """
    strDirTest = os.path.dirname(os.path.abspath(request.module.__file__))
    strDirTmp = str(tmp_path_factory.mktemp(os.path.basename(strDirTest)))
    return TestRunner(strDirTest, strDirTmp, request.module.dicDep)
