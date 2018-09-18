"""Test utility functions."""

import os
from os.path import isfile, join
import numpy as np
from pyprf.analysis import pyprf_main
from pyprf.analysis import utilities as util
from pyprf.analysis.cython_leastsquares_setup_call import setup_cython

# Compile cython code:
setup_cython()

# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def test_main():
    """Run main pyprf function and compare results with template."""
    # --------------------------------------------------------------------------
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

    # Round template reults:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=varRnd)
    aryTmplEcc = np.around(aryTmplEcc.astype(np.float32), decimals=varRnd)
    aryTmplPol = np.around(aryTmplPol.astype(np.float32), decimals=varRnd)
    aryTmplSd = np.around(aryTmplSd.astype(np.float32), decimals=1)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test numpy version

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy.csv')

    # Call main pyprf function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result - R2:
    aryTestNpR2, _, _ = util.load_nii((strDir
                                       + '/result/'
                                       + 'pRF_test_results_np_R2.nii.gz'))

    # Load result - eccentricity:
    aryTestNpEcc, _, _ = util.load_nii((
        strDir + '/result/' + 'pRF_test_results_np_eccentricity.nii.gz'))

    # Load result - polar angle:
    aryTestNpPol, _, _ = util.load_nii(
        (strDir + '/result/' + 'pRF_test_results_np_polar_angle.nii.gz'))

    # Load result - SD:
    aryTestNpSd, _, _ = util.load_nii((strDir
                                       + '/result/'
                                       + 'pRF_test_results_np_SD.nii.gz'))

    # Round test results:
    aryTestNpR2 = np.around(aryTestNpR2.astype(np.float32), decimals=varRnd)
    aryTestNpEcc = np.around(aryTestNpEcc.astype(np.float32), decimals=varRnd)
    aryTestNpPol = np.around(aryTestNpPol.astype(np.float32), decimals=varRnd)
    aryTestNpSd = np.around(aryTestNpSd.astype(np.float32), decimals=1)

    # Test whether the template and test results correspond:
    lgcTestNpR2 = np.all(np.equal(aryTmplR2, aryTestNpR2))
    lgcTestNpEcc = np.all(np.equal(aryTmplEcc, aryTestNpEcc))
    lgcTestNpPol = np.all(np.equal(aryTmplPol, aryTestNpPol))
    lgcTestNpSd = np.all(np.equal(aryTmplSd, aryTestNpSd))

    # Did numpy version pass the test?
    assert lgcTestNpR2
    assert lgcTestNpEcc
    assert lgcTestNpPol
    assert lgcTestNpSd
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython.csv')

    # Call main pyprf function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestCyR2, _, _ = util.load_nii((strDir
                                       + '/result/'
                                       + 'pRF_test_results_cy_R2.nii.gz'))

    # Round test results:
    aryTestCyR2 = np.around(aryTestCyR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestCy = np.all(np.equal(aryTmplR2, aryTestCyR2))

    # Did cython version pass the test?
    assert lgcTestCy
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test tensorflow version

    # Path of config file for tests:
    strCsvCnfgTf = (strDir + '/config_testing_tensorflow.csv')

    # Call main pyprf function:
    pyprf_main.pyprf(strCsvCnfgTf, lgcTest=True)

    # Load result:
    aryTestTfR2, _, _ = util.load_nii((strDir
                                       + '/result/'
                                       + 'pRF_test_results_tf_R2.nii.gz'))

    # Round test results:
    aryTestTfR2 = np.around(aryTestTfR2.astype(np.float32), decimals=varRnd)

    # Test whether the template and test results correspond:
    lgcTestTf = np.all(np.equal(aryTmplR2, aryTestTfR2))

    # Did tensorflow version pass the test?
    assert lgcTestTf
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Clean up

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
    # --------------------------------------------------------------------------

    # assert (lgcTestNp and lgcTestCy and lgcTestTf)


def test_load_large_nii():
    """Test nii-loading function for large nii files."""
    # Load example functional data in normal mode:
    aryFunc01, _, _ = util.load_nii((strDir + '/exmpl_data_func.nii.gz'))

    # Load example functional data in large-file mode:
    aryFunc02, _, _ = util.load_nii((strDir + '/exmpl_data_func.nii.gz'),
                                    varSzeThr=0.0)

    assert np.all(np.equal(aryFunc01, aryFunc02))
