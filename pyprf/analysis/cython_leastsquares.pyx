# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Omer Faruk Gulban & Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# *****************************************************************************
# *** Import modules & adjust cython settings for speedup

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************


# *****************************************************************************
# *** Main function for least squares solution

cpdef np.ndarray[np.float32_t, ndim=1] cy_lst_sq(
    np.ndarray[np.float32_t, ndim=1] vecPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk):
    """
    Cythonised least squares GLM model fitting.

    Parameters
    ----------
    vecPrfTc : np.array
        1D numpy array, at float32 precision, containing a single pRF model
        time course (along time dimension).
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].

    Returns
    -------
    vecRes : np.array
        1D numpy array with model residuals for all voxels in the chunk of
        functional data. Dimensionality: vecRes[voxel]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses. Assumes removal of the mean from
    the functional data and the model. Needs to be compiled before execution
    (see `cython_leastsquares_setup.py`).
    """

    cdef float varVarY = 0
    cdef float[:] vecPrfTc_view = vecPrfTc
    cdef unsigned long varNumVoxChnk, idxVox
    cdef unsigned int idxVol, varNumVols

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])

    # Define 1D array for results (i.e. for residuals of least squares
    # solution):
    cdef np.ndarray[np.float32_t, ndim=1] vecRes = np.zeros(varNumVoxChnk,
                                                            dtype=np.float32)
    # Memory view on array for results:
    cdef float[:] vecRes_view = vecRes

    # Memory view on numpy array with functional data:
    cdef float [:, :] aryFuncChnk_view = aryFuncChnk


    # Calculate variance of pRF model time course (i.e. variance in the model):
    varNumVols = int(vecPrfTc.shape[0])
    for idxVol in range(varNumVols):
        varVarY += vecPrfTc_view[idxVol] ** 2



    # Call optimised cdef function for calculation of residuals:
    vecRes_view = funcCyRes(vecPrfTc_view,
                            aryFuncChnk_view,
                            vecRes_view,
                            varNumVoxChnk,
                            varNumVols,
                            varVarY)

    # Convert memory view to numpy array before returning it:
    vecRes = np.asarray(vecRes_view)

    return vecRes
# *****************************************************************************


# *****************************************************************************
# *** Function for fast calculation of residuals

cdef float[:] funcCyRes(float[:] vecPrfTc_view,
                        float[:, :] aryFuncChnk_view,
                        float[:] vecRes_view,
                        unsigned long varNumVoxChnk,
                        unsigned int varNumVols,
                        float varVarY):

    cdef float varCovXy, varRes, varSlope, varXhat
    cdef unsigned int idxVol
    cdef unsigned long idxVox

    # Loop through voxels:
    for idxVox in range(varNumVoxChnk):

        # Covariance and residuals of current voxel:
        varCovXy = 0
        varRes = 0

        # Loop through volumes and calculate covariance between the model and
        # the current voxel:
        for idxVol in range(varNumVols):
            varCovXy += (aryFuncChnk_view[idxVol, idxVox]
                         * vecPrfTc_view[idxVol])
        # Obtain the slope of the regression of the model on the data:
        varSlope = varCovXy / varVarY

        # Loop through volumes again in order to calculate the error in the
        # prediction:
        for idxVol in range(varNumVols):
            # The predicted voxel time course value:
            varXhat = vecPrfTc_view[idxVol] * varSlope
            # Mismatch between prediction and actual voxel value (variance):
            varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

        vecRes_view[idxVox] = varRes

    # Return memory view:
    return vecRes_view
# *****************************************************************************
