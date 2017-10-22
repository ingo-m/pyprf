"""Test utility functions."""

import os
from os.path import isfile, join
import numpy as np
from .. import pyprf_main
from .. import utilities
from cython_leastsquares_setup_call import setup_cython


setup_cython()

# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def test_main():
    """Run main pyprf function and compare results with template."""
    # --------------------------------------------------------------------------
    # *** Test numpy version

    # Path of config file for tests:
    strCsvCnfgNp = (strDir + '/config_testing_numpy.csv')

    # Call main pyprf function:
    pyprf_main.pyprf(strCsvCnfgNp, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = utilities.load_nii((strDir
                                          + '/result/pRF_test_results_R2.nii'))

    # Load result templates:
    aryTmplR2, _, _ = utilities.load_nii((strDir
                                          + '/exmpl_data_results_R2.nii.gz'))

    # Round template and test results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=4)
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=4)

    # Test whether the template and test results correspond:
    lgcTestNp = np.all(np.equal(aryTmplR2, aryTestR2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # *** Test cython version

    # Path of config file for tests:
    strCsvCnfgCy = (strDir + '/config_testing_cython.csv')

    # Call main pyprf function:
    pyprf_main.pyprf(strCsvCnfgCy, lgcTest=True)

    # Load result:
    aryTestR2, _, _ = utilities.load_nii((strDir
                                          + '/result/pRF_test_results_R2.nii'))

    # Load result templates:
    aryTmplR2, _, _ = utilities.load_nii((strDir
                                          + '/exmpl_data_results_R2.nii.gz'))

    # Round template and test results:
    aryTmplR2 = np.around(aryTmplR2.astype(np.float32), decimals=3)
    aryTestR2 = np.around(aryTestR2.astype(np.float32), decimals=3)

    # Test whether the template and test results correspond:
    lgcTestCy = np.all(np.equal(aryTmplR2, aryTestR2))
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

    assert (lgcTestNp and lgcTestCy)
