"""
Test creation of design matrices for the stimulus presentation.

The stimulus presentation code (`pyprf/stimulus_presentation`) is not part of
the pyprf python package; it is distributed as a folder that is run from
PsychoPy. Therefore, its tests are not inside that folder, and the code is
loaded from its file path. PsychoPy is not needed for these tests.
"""

import os
import importlib.util

import numpy as np
import pytest


# Path of the design matrix script:
strPthScrpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           'pyprf', 'stimulus_presentation', 'code',
                           'create_design_matrix.py')

objSpec = importlib.util.spec_from_file_location('create_design_matrix',
                                                 strPthScrpt)
create_design_matrix = importlib.util.module_from_spec(objSpec)
objSpec.loader.exec_module(create_design_matrix)


def get_param(strPth, **kwargs):
    """Design matrix parameters (default values of the GUI)."""
    dicParam = {'Output file name': 'Run_01',
                'Output path': strPth,
                'TR [s]': 2.079,
                'Number of bar orientations': 4,
                'Number of bar positions on x-axis': 14,
                'Number of bar positions on y-axis': 8,
                'Number of blocks': 4,
                'Number of rest trials': 1,
                'Inter-trial interval for targets [s]': 15.0,
                'Initial rest period [volumes]': 10,
                'Final rest period [volumes]': 10,
                'Full screen:': True,
                'Stimulus contrasts': [1.0]}
    dicParam.update(kwargs)
    return dicParam


@pytest.mark.parametrize('lgcFull', [True, False])
def test_crt_design(tmp_path, lgcFull):
    """Create a design matrix and check its properties."""
    np.random.seed(0)
    dicParam = get_param(str(tmp_path), **{'Full screen:': lgcFull})
    create_design_matrix.crt_design(dicParam)

    # Design matrix is saved as npz file, and in human-readable format:
    assert (tmp_path / 'Run_01.npz').is_file()
    assert (tmp_path / 'Run_01.txt').is_file()

    objNpz = np.load(str(tmp_path / 'Run_01.npz'))
    aryDsg = objNpz['aryDsg']
    vecTrgt = objNpz['vecTrgt']
    varNumVol = int(objNpz['varNumVol'])
    varTr = float(objNpz['varTr'])

    assert varTr == 2.079
    assert bool(objNpz['lgcFull']) == lgcFull
    assert aryDsg.shape == (varNumVol, 4)

    # Initial and final rest periods (10 volumes each):
    assert np.all(aryDsg[:10, :] == 0.0)
    assert np.all(aryDsg[-10:, :] == 0.0)

    # Stimulus volumes: 4 blocks x 4 orientations x 14 positions (in full
    # screen mode, without horizontal bars outside of the screen), plus one
    # rest block of 3 volumes:
    aryStim = aryDsg[aryDsg[:, 0] == 1.0, :]
    varNumHor = np.unique(aryStim[aryStim[:, 2] == 0.0, 1]).shape[0]
    assert aryStim.shape[0] == 4 * (3 * 14 + varNumHor)
    assert varNumHor == (9 if lgcFull else 14)
    assert varNumVol == aryStim.shape[0] + 10 + 10 + 3
    assert set(np.unique(aryStim[:, 2])) == {0.0, 45.0, 90.0, 135.0}

    # Target events are between 3 and 35 seconds apart, and are not in the
    # initial & final rest periods:
    assert np.all(np.diff(vecTrgt) >= 3.0)
    assert np.all(np.diff(vecTrgt) <= 35.0)
    assert vecTrgt[0] >= 10 * varTr
    assert vecTrgt[-1] <= (varNumVol - 10) * varTr


@pytest.mark.xfail(strict=True, reason=(
    'Known bug: in full screen mode, horizontal bars are kept at positions 3 '
    'to 11 (instead of 3 to 10), so the uppermost one is outside of the '
    'screen.'))
def test_full_screen_horizontal_positions(tmp_path):
    """In full screen mode, horizontal bars are centred on the screen."""
    np.random.seed(0)
    create_design_matrix.crt_design(get_param(str(tmp_path)))
    aryDsg = np.load(str(tmp_path / 'Run_01.npz'))['aryDsg']
    aryStim = aryDsg[aryDsg[:, 0] == 1.0, :]

    # With 14 positions on the x-axis and 8 on the y-axis, the horizontal bars
    # should be at the 8 central positions (3 to 10), symmetric around the
    # centre of the screen (6.5):
    vecPosHor = np.unique(aryStim[aryStim[:, 2] == 0.0, 1])
    assert list(vecPosHor) == list(range(3, 11))


def test_crt_design_two_orientations(tmp_path):
    """Design matrix with only horizontal and vertical bars."""
    np.random.seed(0)
    dicParam = get_param(str(tmp_path),
                         **{'Number of bar orientations': 2,
                            'Full screen:': False,
                            'Stimulus contrasts': [0.05, 1.0]})
    create_design_matrix.crt_design(dicParam)

    aryDsg = np.load(str(tmp_path / 'Run_01.npz'))['aryDsg']
    aryStim = aryDsg[aryDsg[:, 0] == 1.0, :]

    assert set(np.unique(aryStim[:, 2])) == {0.0, 90.0}
    assert set(np.unique(aryStim[:, 3])) == {0.05, 1.0}
    # 4 blocks x 2 orientations x 14 positions x 2 contrasts:
    assert aryStim.shape[0] == 4 * 2 * 14 * 2


def test_crt_design_existing_file(tmp_path):
    """An existing design matrix is only overwritten if requested."""
    dicParam = get_param(str(tmp_path))
    strPthNpz = create_design_matrix.crt_design(dicParam)
    assert strPthNpz == str(tmp_path / 'Run_01.npz')
    varMtime = os.path.getmtime(strPthNpz)

    with pytest.raises(FileExistsError):
        create_design_matrix.crt_design(dicParam)
    assert os.path.getmtime(strPthNpz) == varMtime

    create_design_matrix.crt_design(dicParam, lgcOvwr=True)
    assert os.path.getmtime(strPthNpz) >= varMtime


def test_func_free_name(tmp_path):
    """The default name of a new design matrix is the first unused name."""
    assert create_design_matrix.func_free_name(str(tmp_path)) == 'Run_01'
    for strNme in ['Run_01', 'Run_02', 'Run_04']:
        create_design_matrix.crt_design(
            get_param(str(tmp_path), **{'Output file name': strNme}))
    assert create_design_matrix.func_free_name(str(tmp_path)) == 'Run_03'
