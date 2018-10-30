"""Test utility functions."""


import os
from os.path import isfile, join
import numpy as np

from pyprf.analysis import pyprf_main
from pyprf.analysis import utilities as util


# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def test_main():
    """Run main pyprf function and compare results with template."""
    # -------------------------------------------------------------------------
    # *** Preparations

    # Decimal places to round before comparing template and test results:
    varRnd = 3

    # Load template result - R2:
    aryTmplR2, _, _ = util.load_nii((strDir
                                     + '/exmpl_data_results_R2.nii.gz'))

    # Load template result - eccentricity:
    aryTmplEcc, _, _ = util.load_nii(
        (strDir + '/exmpl_data_results_eccentricity.nii.gz'))

    # Load template result - polar angle:
    aryTmplPol, _, _ = util.load_nii(
        (strDir + '/exmpl_data_results_polar_angle.nii.gz'))

    # Load template result - SD:
    aryTmplSd, _, _ = util.load_nii((strDir
                                     + '/exmpl_data_results_SD.nii.gz'))

    # Load template result - parameter estimate condition 1:
    aryTmplPe01, _, _ = util.load_nii((strDir
                                      + '/exmpl_data_results_PE_01.nii.gz'))

    # Load template result - parameter estimate condition 2:
    aryTmplPe02, _, _ = util.load_nii((strDir
                                      + '/exmpl_data_results_PE_02.nii.gz'))

    # Round template reults:
    aryTmplR2 = np.around(aryTmplR2, decimals=varRnd).astype(np.float32)
    aryTmplEcc = np.around(aryTmplEcc, decimals=varRnd).astype(np.float32)
    aryTmplPol = np.around(aryTmplPol, decimals=varRnd).astype(np.float32)
    aryTmplSd = np.around(aryTmplSd, decimals=varRnd).astype(np.float32)
    aryTmplPe01 = np.around(aryTmplPe01, decimals=varRnd).astype(np.float32)
    aryTmplPe02 = np.around(aryTmplPe02, decimals=varRnd).astype(np.float32)

    # -------------------------------------------------------------------------
    # *** Test pyprf main pipeline

    # Test numpy, cython, and tensorflow version. List with version
    # abbreviations:
    lstVrsn = ['np', 'cy', 'cy_hdf5']  # 'np_hdf5'

    # Path of config file for tests (version abbreviation left open):
    strCsvCnfg = (strDir + '/config_testing_{}.csv')

    for strVrsn in lstVrsn:

        # Call main pyprf function:
        pyprf_main.pyprf(strCsvCnfg.format(strVrsn), lgcTest=True)

        # Load result - R2:
        aryTestR2, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_R2.nii.gz'.format(strVrsn)))

        # Load result - eccentricity:
        aryTestEcc, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_eccentricity.nii.gz'.format(strVrsn)))

        # Load result - polar angle:
        aryTestPol, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_polar_angle.nii.gz'.format(strVrsn)))

        # Load result - SD:
        aryTestSd, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_SD.nii.gz'.format(strVrsn)))

        # Load result - parameter estimate condition 1:
        aryTestlPe01, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_PE_01.nii.gz'.format(strVrsn)))

        # Load result - parameter estimate condition 2:
        aryTestlPe02, _, _ = util.load_nii(
            (strDir + '/result/'
             + 'pRF_test_results_{}_PE_02.nii.gz'.format(strVrsn)))

        # Round test results:
        aryTestR2 = np.around(aryTestR2, decimals=varRnd).astype(np.float32)
        aryTestEcc = np.around(aryTestEcc, decimals=varRnd).astype(np.float32)
        aryTestPol = np.around(aryTestPol, decimals=varRnd).astype(np.float32)
        aryTestSd = np.around(aryTestSd, decimals=varRnd).astype(np.float32)
        aryTestlPe01 = np.around(aryTestlPe01, decimals=varRnd
                                 ).astype(np.float32)
        aryTestlPe02 = np.around(aryTestlPe02, decimals=varRnd
                                 ).astype(np.float32)

        # Test whether the template and test results correspond:
        # print('np.max(np.abs(np.subtract(aryTmplR2, aryTestR2)))')
        # print(np.max(np.abs(np.subtract(aryTmplR2, aryTestR2))))
        lgcTestR2 = np.all(np.equal(aryTmplR2, aryTestR2))
        lgcTestEcc = np.all(np.equal(aryTmplEcc, aryTestEcc))
        lgcTestPol = np.all(np.equal(aryTmplPol, aryTestPol))
        lgcTestSd = np.all(np.equal(aryTmplSd, aryTestSd))
        lgcTestPe01 = np.all(np.equal(aryTmplPe01, aryTestlPe01))
        lgcTestPe02 = np.all(np.equal(aryTmplPe02, aryTestlPe02))

        # Did version pass the test?
        assert lgcTestR2
        assert lgcTestEcc
        assert lgcTestPol
        assert lgcTestSd
        assert lgcTestPe01
        assert lgcTestPe02

    # -------------------------------------------------------------------------
    # *** Clean up testing results

    # Path of directory with results:
    strDirRes = strDir + '/result/'

    # Get list of files in results directory:
    lstFls = [f for f in os.listdir(strDirRes) if isfile(join(strDirRes, f))]

    # Delete results of test:
    for strTmp in lstFls:
        if '.nii' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))
        elif '.npy' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))
        elif '.hdf5' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))

    # -------------------------------------------------------------------------
    # *** Clean up intermediate results (hdf5 files)

    # Path of directory with time courses converted to hdf5:
    strDirRes = strDir + '/'

    # Get list of files in results directory:
    lstFls = [f for f in os.listdir(strDirRes) if isfile(join(strDirRes, f))]

    # Delete results of test:
    for strTmp in lstFls:
        if '.hdf5' in strTmp:
            # print(strTmp)
            os.remove((strDirRes + '/' + strTmp))
    # -------------------------------------------------------------------------
