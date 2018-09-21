# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting with 2 predictors."""

# Part of pyprf_feature library
# Copyright (C) 2018  Omer Faruk Gulban & Ingo Marquardt & Marian Schneider
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

import time
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************


# *****************************************************************************
# *** Main function least squares solution, no cross-validation, 2 predictors

cpdef tuple cy_lst_sq_two(
    np.ndarray[np.float32_t, ndim=2] aryPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk):
    """
    Cythonised least squares GLM model fitting.

    Parameters
    ----------
    aryPrfTc : np.array
        2D numpy array, at float32 precision, containing two pRF model
        time courses as two columns. Dimensionality: aryPrfTc[time, 2].
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].

    Returns
    -------
    vecPe : np.array
        2D numpy array with parameter estimates for all voxels in the chunk of
        functional data. Dimensionality: vecPe[2, voxel]
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
    ### Timing
    varTme01 = time.time()
    cdef:
        float varVarX1, varVarX2, varVarX1X2
        unsigned long varNumVoxChnk, idxVox
        unsigned int idxVol, varNumVols

    # Initial variances and covariances
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
    cdef np.ndarray[np.float32_t, ndim=2] vecPe = np.zeros((varNumVoxChnk, 2),
                                                           dtype=np.float32)

    # Memory view on array for results:
    cdef float[:] vecRes_view = vecRes

    # Memory view on array for parameter estimates:
    cdef float[:, :] vecPe_view = vecPe

    # Memory view on predictor time courses:
    cdef float[:, :] aryPrfTc_view = aryPrfTc

    # Memory view on numpy array with functional data:
    cdef float[:, :] aryFuncChnk_view = aryFuncChnk

    # Calculate variance of pRF model time course (i.e. variance in the model):
    varNumVols = int(aryPrfTc.shape[0])

    # get the variance for x1
    for idxVol in range(varNumVols):
        varVarX1 += aryPrfTc_view[idxVol, 0] ** 2
        varVarX2 += aryPrfTc_view[idxVol, 1] ** 2
        varVarX1X2 += aryPrfTc_view[idxVol, 0] * aryPrfTc_view[idxVol, 1]

    # Call optimised cdef function for calculation of residuals:
    vecRes_view, vecPe_view = func_cy_res_two(aryPrfTc_view,
                                              aryFuncChnk_view,
                                              vecRes_view,
                                              vecPe_view,
                                              varNumVoxChnk,
                                              varNumVols,
                                              varVarX1,
                                              varVarX2,
                                              varVarX1X2)

    # Convert memory view to numpy array before returning it:
    vecRes = np.asarray(vecRes_view)
    vecPe = np.asarray(vecPe_view).T

    return vecPe, vecRes


# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, no cross-validation, 2 predictors

cdef (float[:], float[:, :]) func_cy_res_two(float[:, :] aryPrfTc_view,
                                             float[:, :] aryFuncChnk_view,
                                             float[:] vecRes_view,
                                             float[:, :] vecPe_view,
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
                          * aryPrfTc_view[idxVol, 0])
            varCovX2y += (aryFuncChnk_view[idxVol, idxVox]
                          * aryPrfTc_view[idxVol, 1])
        # calculate denominator
        varDen = varVarX1 * varVarX2 - varVarX1X2 ** 2
        # Obtain the slope of the regression of the model on the data:
        varSlope1 = (varVarX2 * varCovX1y - varVarX1X2 * varCovX2y) / varDen
        varSlope2 = (varVarX1 * varCovX2y - varVarX1X2 * varCovX1y) / varDen

        # Loop through volumes again in order to calculate the error in the
        # prediction:
        for idxVol in range(varNumVols):
            # The predicted voxel time course value:
            varXhat = (aryPrfTc_view[idxVol, 0] * varSlope1 +
                       aryPrfTc_view[idxVol, 1] * varSlope2)
            # Mismatch between prediction and actual voxel value (variance):
            varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

        vecRes_view[idxVox] = varRes
        vecPe_view[idxVox, 0] = varSlope1
        vecPe_view[idxVox, 1] = varSlope2

    # Return memory view:
    return vecRes_view, vecPe_view
# *****************************************************************************

# *****************************************************************************
# *** Main function least squares solution, with cross-validation, 2 predictors

cpdef np.ndarray[np.float32_t, ndim=2] cy_lst_sq_xval_two(
    np.ndarray[np.float32_t, ndim=2] vecPrfTc,
    np.ndarray[np.float32_t, ndim=2] aryFuncChnk,
    np.ndarray[np.int32_t, ndim=2] aryIdxTrn,
    np.ndarray[np.int32_t, ndim=2] aryIdxTst
    ):
    """
    Cythonised least squares GLM model fitting with cross validation.

    Parameters
    ----------
    vecPrfTc : np.array
        2D numpy array, at float32 precision, containing a single pRF model
        time course (along time dimension).
    aryFuncChnk : np.array
        2D numpy array, at float32 precision, containing a chunk of functional
        data (i.e. voxel time courses). Dimensionality: aryFuncChnk[time,
        voxel].
    aryIdxTrn : np.array
        2D numpy array, at int32 precision, containing a trainings indices for
        cross-validation.
    aryIdxTst : np.array
        2D numpy array, at int32 precision, containing a test indices for
        cross-validation.

    Returns
    -------
    aryResXval : np.array
        2D numpy array with cross validation error for all voxels in the chunk of
        functional data and all cross validation folds.
        Dimensionality: aryResXval[voxel, varNumXval]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses with k-fold cross validation.
    Assumes removal of the mean from the functional data and the model.
    Needs to be compiled before execution (see `cython_leastsquares_setup.py`).
    """
    cdef:
        float[:, :] aryPrfTc_view = vecPrfTc
        float [:, :] aryFuncChnk_view = aryFuncChnk
        int [:, :] aryIdxTrn_view = aryIdxTrn
        int [:, :] aryIdxTst_view = aryIdxTst
        unsigned long varNumVoxChnk, idxVox
        unsigned int idxVol, idxXval, varNumXval, varNumVolTrn, varNumVolTst
        int[:] vecIdxTrn

    # Number of voxels in the input data chunk:
    varNumVoxChnk = int(aryFuncChnk.shape[1])
    # Number of cross-validations:
    varNumXval = int(aryIdxTrn.shape[1])
    # Number of training volumes
    varNumVolTrn = int(aryIdxTrn.shape[0])
    # Number of testing volumes
    varNumVolTst = int(aryIdxTst.shape[0])

    # Define 2D array for residuals (here crossvalidation error) of least
    # squares solution), initialized with all zeros here:
    cdef np.ndarray[np.float32_t, ndim=2] aryResXval = np.zeros((varNumVoxChnk,
                                                                 varNumXval),
                                                                dtype=np.float32)

    # Memory view on array for residuals (here crossvalidation error)
    cdef float[:, :] aryResXval_view = aryResXval

    # Define 1D arrays for co/variances in training model time courses across
    # folds, initialized with all zeros here
    cdef np.ndarray[np.float32_t, ndim=1] vecVarX1 = np.zeros(varNumXval,
                                                              dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] vecVarX2 = np.zeros(varNumXval,
                                                              dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] vecVarXY = np.zeros(varNumXval,
                                                              dtype=np.float32)
    # Memory view on array for co/variances in training model time courses:
    cdef float[:] vecVarX1_view = vecVarX1
    cdef float[:] vecVarX2_view = vecVarX2
    cdef float[:] vecVarXY_view = vecVarXY

    # Calculate variance of training pRF model time course (i.e. variance in
    # the model) - separately for every fold:
    for idxXval in range(varNumXval):
        # get vector with volumes for training
        vecIdxTrn = aryIdxTrn_view[:, idxXval]
        for idxVol in vecIdxTrn:
            vecVarX1_view[idxXval] += aryPrfTc_view[idxVol, 0] ** 2
            vecVarX2_view[idxXval] += aryPrfTc_view[idxVol, 1] ** 2
            vecVarXY_view[idxXval] += (aryPrfTc_view[idxVol, 0] *
                                       aryPrfTc_view[idxVol, 1])

    # Call optimised cdef function for calculation of residuals:
    aryResXval_view = func_cy_res_xval(aryPrfTc_view,
                                       aryFuncChnk_view,
                                       aryIdxTrn_view,
                                       aryIdxTst_view,
                                       aryResXval_view,
                                       varNumXval,
                                       varNumVoxChnk,
                                       varNumVolTrn,
                                       varNumVolTst,
                                       vecVarX1_view,
                                       vecVarX2_view,
                                       vecVarXY_view)

    # Convert memory view to numpy array before returning it:
    aryResXval = np.asarray(aryResXval_view)

    return aryResXval

# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, with cross-validation, 1 predictor

cdef float[:, :] func_cy_res_xval(float[:, :] aryPrfTc_view,
                                  float[:, :] aryFuncChnk_view,
                                  int[:, :] aryIdxTrn_view,
                                  int[:, :] aryIdxTst_view,
                                  float[:, :] aryResXval_view,
                                  unsigned int varNumXval,
                                  unsigned long varNumVoxChnk,
                                  unsigned int varNumVolTrn,
                                  unsigned int varNumVolTst,
                                  float[:] vecVarX1_view,
                                  float[:] vecVarX2_view,
                                  float[:] vecVarXY_view):

    cdef:
        float varCovX1y, varCovX2y, varRes
        float varVarX1, varVarX2, varVarX1X2
        float varSlope1, varSlope2, varXhat, varDen
        unsigned int idxVol, idxXval, idxItr
        unsigned long idxVox

    # Loop through cross-validations
    for idxXval in range(varNumXval):

        # Loop through voxels:
        for idxVox in range(varNumVoxChnk):

            # Covariance and residuals of current voxel:
            varCovX1y = 0
            varCovX2y = 0
            varRes = 0

            # Loop through trainings volumes and calculate covariance between
            # the training model and the current voxel:
            for idxItr in range(varNumVolTrn):
                # get the training volume
                idxVol = aryIdxTrn_view[idxItr, idxXval]
                
                varCovX1y += (aryFuncChnk_view[idxVol, idxVox]
                              * aryPrfTc_view[idxVol, 0])
                varCovX2y += (aryFuncChnk_view[idxVol, idxVox]
                              * aryPrfTc_view[idxVol, 1])

            # Get the variance of the training model time courses for this fold
            varVarX1 = vecVarX1_view[idxXval]
            varVarX2 = vecVarX2_view[idxXval]
            varVarX1X2 = vecVarXY_view[idxXval]

            # calculate denominator
            varDen = varVarX1 * varVarX2 - varVarX1X2 ** 2
            # Obtain the slope of the regression of the model on the data:
            varSlope1 = ((varVarX2 * varCovX1y - varVarX1X2 * varCovX2y) /
                         varDen)
            varSlope2 = ((varVarX1 * varCovX2y - varVarX1X2 * varCovX1y) /
                         varDen)

            # Loop through test volumes and calculate the predicted time course
            # value and the mismatch between prediction and actual voxel value
            for idxItr in range(varNumVolTst):
                # get the test volume
                idxVol = aryIdxTst_view[idxItr, idxXval]
                # The predicted voxel time course value:
                varXhat = (aryPrfTc_view[idxVol, 0] * varSlope1 +
                           aryPrfTc_view[idxVol, 1] * varSlope2)
                # Mismatch between prediction and actual vxl value (variance):
                varRes += (aryFuncChnk_view[idxVol, idxVox] - varXhat) ** 2

            aryResXval_view[idxVox, idxXval] = varRes

    # Return memory view
    return aryResXval_view

# *****************************************************************************