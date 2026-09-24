"""Test pyprf main pipeline and utility functions."""


import os
import numpy as np
import pytest

from pyprf.analysis import utilities as util
from pyprf.analysis.load_config import load_config


# Get directory of this file:
strDir = os.path.dirname(os.path.abspath(__file__))

# Version abbreviation -> version that has to run first, because it creates the
# pRF model time courses (see `TestRunner` in `/conftest.py`):
dicDep = {'cy': 'np', 'np_hdf5': 'cy_hdf5'}


@pytest.mark.parametrize('strVrsn', ['np', 'cy', 'cy_hdf5', 'np_hdf5'])
def test_main(test_runner, strVrsn):
    """Run main pyprf function and compare results with template."""
    test_runner.run(strVrsn)
    test_runner.assert_results(strVrsn,
                               ['R2', 'eccentricity', 'polar_angle', 'SD'])


def test_load_large_nii():
    """Test nii-loading function for large nii files."""
    # Load example functional data in normal mode:
    aryFunc01, _, _ = util.load_nii((strDir + '/exmpl_data_func_3vols.nii.gz'))

    # Load example functional data in large-file mode:
    aryFunc02, _, _ = util.load_nii((strDir + '/exmpl_data_func_3vols.nii.gz'),
                                    varSzeThr=0.0)

    assert np.all(np.equal(aryFunc01, aryFunc02))


def test_unsupported_version(tmp_path):
    """Test that the removed GPU version gives an informative error."""
    with open(strDir + '/config_testing_np.csv', 'r') as objFle:
        strCnfg = objFle.read()
    strCnfg = strCnfg.replace("strVersion = 'numpy'", "strVersion = 'gpu'")
    strCsvCnfg = str(tmp_path / 'config_gpu.csv')
    with open(strCsvCnfg, 'w') as objFle:
        objFle.write(strCnfg)

    with pytest.raises(ValueError, match="'gpu' version was removed"):
        load_config(strCsvCnfg)
