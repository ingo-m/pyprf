"""Test pyprf main pipeline with two predictors (e.g. two contrast levels)."""


import pytest


# Version abbreviation -> version that has to run first, because it creates the
# pRF model time courses (see `TestRunner` in `/conftest.py`):
dicDep = {'cy': 'np'}


@pytest.mark.parametrize('strVrsn', ['np', 'cy', 'cy_hdf5'])
def test_main(test_runner, strVrsn):
    """Run main pyprf function and compare results with template."""
    test_runner.run(strVrsn)
    test_runner.assert_results(strVrsn,
                               ['R2', 'eccentricity', 'polar_angle', 'SD',
                                'PE_01', 'PE_02'])
