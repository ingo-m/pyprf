# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting with 2 predictors."""

# Part of pyprf library
# Copyright (C) 2018  Omer Faruk Gulban & Ingo Marquardt & Marian Schneider
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


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
# *** Main function least squares solution, two predictors

cpdef tuple cy_lst_sq_two(
    np.ndarray[np.float32_t, ndim=2] aryPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk):
    """
    Cythonised least squares GLM model fitting with two predictors.

    Parameters
    ----------
    aryPrfTc : np.array
        2D numpy array, at float32 precision, containing two pRF model time
        courses as two columns. E.g. model time courses for one pRF position &
        size, for two stimulus conditions (such as luminance contrast).
        Dimensionality: aryPrfTc[2, time].
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].

    Returns
    -------
    vecRes : np.array
        1D numpy array with model residuals for all voxels in the chunk of
        functional data. Dimensionality: vecRes[voxel]
    aryPe : np.array
        2D numpy array with parameter estimates for all voxels in the chunk of
        functional data. Dimensionality: aryPe[2, voxel]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses. Assumes removal of the mean from
    the functional data and the model. Needs to be compiled before execution
    (see `cython_leastsquares_setup.py`).

    """
    cdef:
        float varVarX1, varVarX2, varVarX1X2
        unsigned long varNumVoxChnk, idxVox
        unsigned int idxVol, varNumVols

    # Initial variances and covariance:
    varVarX1 = 0
    varVarX2 = 0
    varVarX1X2 = 0

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])

    # Define 1D array for results (i.e. for residuals of least squares
    # solution):
    cdef np.ndarray[np.float32_t, ndim=1] vecRes = np.zeros(varNumVoxChnk,
                                                            dtype=np.float32)

    # Define 2D array for results - parameter estimate:
    cdef np.ndarray[np.float32_t, ndim=2] aryPe = np.zeros((varNumVoxChnk, 2),
                                                           dtype=np.float32)

    # Memory view on array for results:
    cdef float[:] vecRes_view = vecRes

    # Memory view on array for parameter estimates:
    cdef float[:, :] aryPe_view = aryPe

    # Memory view on predictor time courses:
    cdef float[:, :] aryPrfTc_view = aryPrfTc

    # Memory view on numpy array with functional data:
    cdef float[:, :] aryFuncChnk_view = aryFuncChnk

    # Calculate variance of pRF model time course (i.e. variance in the model):
    varNumVols = int(aryPrfTc.shape[1])

    # Calculate variances and covariances of the two pRF model time courses:
    for idxVol in range(varNumVols):
        varVarX1 += aryPrfTc_view[0, idxVol] ** 2
        varVarX2 += aryPrfTc_view[1, idxVol] ** 2
        varVarX1X2 += aryPrfTc_view[0, idxVol] * aryPrfTc_view[1, idxVol]

    # Call optimised cdef function for calculation of residuals:
    vecRes_view, aryPe_view = func_cy_res_two(aryPrfTc_view,
                                              aryFuncChnk_view,
                                              vecRes_view,
                                              aryPe_view,
                                              varNumVoxChnk,
                                              varNumVols,
                                              varVarX1,
                                              varVarX2,
                                              varVarX1X2)

    # Convert memory view to numpy array before returning it:
    vecRes = np.asarray(vecRes_view)
    aryPe = np.asarray(aryPe_view).T

    return vecRes, aryPe
# *****************************************************************************

# *****************************************************************************
# *** Fast calculation residuals, two predictors

cdef (float[:], float[:, :]) func_cy_res_two(float[:, :] aryPrfTc_view,
                                             float[:, :] aryFuncChnk_view,
                                             float[:] vecRes_view,
                                             float[:, :] aryPe_view,
                                             unsigned long varNumVoxChnk,
                                             unsigned int varNumVols,
                                             float varVarX1,
                                             float varVarX2,
                                             float varVarX1X2):

    cdef:
        float varCovX1y, varCovX2y, varRes
        float varDen, varSlope1, varSlope2, varXhat
        unsigned int idxVol
        unsigned long idxVox

    # Calculate denominator:
    varDen = varVarX1 * varVarX2 - varVarX1X2 ** 2

    # Loop through voxels:
    for idxVox in range(varNumVoxChnk):

        # Covariance and residuals of current voxel:
        varCovX1y = 0
        varCovX2y = 0
        varRes = 0

        # Loop through volumes and calculate covariance between the model and
        # the current voxel:
        for idxVol in range(varNumVols):
            varCovX1y += (aryFuncChnk_view[idxVol, idxVox]
                          * aryPrfTc_view[0, idxVol])
            varCovX2y += (aryFuncChnk_view[idxVol, idxVox]
                          * aryPrfTc_view[1, idxVol])

        # Obtain the slope of the regression of the model on the data:
        varSlope1 = (varVarX2 * varCovX1y - varVarX1X2 * varCovX2y) / varDen
        varSlope2 = (varVarX1 * varCovX2y - varVarX1X2 * varCovX1y) / varDen

        # Loop through volumes again in order to calculate the error in the
        # prediction:
        for idxVol in range(varNumVols):
            # The predicted voxel time course value:
            varXhat = (aryPrfTc_view[0, idxVol] * varSlope1
                       + aryPrfTc_view[1, idxVol] * varSlope2)
            # Mismatch between prediction and actual voxel value (variance):
            varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

        vecRes_view[idxVox] = varRes
        aryPe_view[idxVox, 0] = varSlope1
        aryPe_view[idxVox, 1] = varSlope2

    # Return memory view:
    return vecRes_view, aryPe_view
# *****************************************************************************
