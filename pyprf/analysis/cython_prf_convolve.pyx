# -*- coding: utf-8 -*-
"""Cythonised pRF model convolution."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
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
from libc.math cimport pow, sqrt, exp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************


# *****************************************************************************
# *** Python-cython interface

cpdef np.ndarray[np.float32_t, ndim=3] prf_conv(
    np.ndarray[np.float32_t, ndim=2] aryX,
    np.ndarray[np.float32_t, ndim=2] aryY,
    np.ndarray[np.float32_t, ndim=2] aryMdlParamsChnk,
    np.ndarray[np.float32_t, ndim=4] aryPixConv):
    """
    Cythonised calculation of pRF model time courses.

    Parameters
    ----------
    aryX : np.array
        2D array meshgrid array, representing x dimension of the visual field.
        Can be created with `scipy.mgrid`.
    aryY : np.array
        2D array meshgrid array, representing y dimension of the visual field.
        Can be created with `scipy.mgrid`.
    aryMdlParamsChnk : np.array
        2D numpy array containing the parameters for the pRF models to be
        created. Dimensionality: `aryMdlParamsChnk[model-ID, parameter-value]`.
        For each model there are four values: (0) an index starting from zero,
        (1) the x-position, (2) the y-position, and (3) the standard deviation.
        Parameters 1, 2 , and 3 are in units of the upsampled visual space.
    aryPixConv : np.array
        4D numpy array containing the pixel-wise, HRF-convolved design matrix,
        with the following structure: `aryPixConv[x-pixels, y-pixels,
        conditions, volumes]`.

    Returns
    -------
    aryPrfTc : np.array
        3D numpy array with pRF model time courses, shape:
        aryPrfTc[varNumMdls, varNumCon, varNumVol].

    Notes
    -----
    Cythonised 2D convolution of model time courses with Gaussian pRF model.
    """
    cdef unsigned int varNumX, varNumY
    cdef unsigned long varNumMdls, varNumCon, varNumVol

    # Memory view on pixel-wise, HRF-convolved design matrix (shape:
    # aryPixConv[x-pixels, y-pixels, conditions, volumes]).
    cdef float [:, :, :, :] aryPixConv_view = aryPixConv.astype(np.float32)

    # Memory view on x and y dimension meshgrids:
    cdef float [:, :] aryX_view = aryX.astype(np.float32)
    cdef float [:, :] aryY_view = aryY.astype(np.float32)

    # Number of models (i.e., number of combinations of model parameters) in
    # the input data chunk:
    varNumMdls = int(aryMdlParamsChnk.shape[0])

    # Number of pixels in X and Y dimensions:
    varNumX = int(aryPixConv.shape[0])
    varNumY = int(aryPixConv.shape[1])

    # Number of conditions:
    varNumCon = int(aryPixConv.shape[2])

    # Number of volumes:
    varNumVol = int(aryPixConv.shape[3])

    # Array for result - 3D numpy array with pRF model time courses:
    cdef np.ndarray[np.float32_t, ndim=3] aryPrfTc = np.zeros((varNumMdls,
                                                               varNumCon,
                                                               varNumVol),
                                                              dtype=np.float32)

    # Memory view on array for results (convolved pRF time courses):
    cdef float[:, :, :] aryPrfTc_view = aryPrfTc

    # Array for Gaussian models (will be updated on each iteration):
    cdef np.ndarray[np.float32_t, ndim=2] aryGauss = np.zeros((varNumX,
                                                               varNumY),
                                                              dtype=np.float32)

    # Memory view on array for Gaussian models:
    cdef float[:, :] aryGauss_view = aryGauss

    # Array for intermediate results (will be updated on each iteration):
    cdef np.ndarray[np.float32_t, ndim=3] aryTmp = np.zeros((varNumX,
                                                             varNumY,
                                                             varNumVol),
                                                            dtype=np.float32)

    # Memory view on array for Gaussian models:
    cdef float[:, :, :] aryTmp_view = aryTmp

    # Call optimised cdef function for calculation of residuals:
    aryPrfTc_view = cy_prf_conv(aryX_view,
                                aryY_view,
                                aryMdlParams_view,
                                aryPixConv_view,
                                aryPrfTc_view,
                                aryGauss_view,
                                aryTmp_view,
                                varNumMdls,
                                varNumCon,
                                varNumVol,
                                varNumX,
                                varNumY)

    # Convert memory view to numpy array before returning it. Shape:
    # aryPrfTc[idxMdl, idxCon, idxVol].
    aryPrfTc = np.asarray(aryPrfTc_view)

    return aryPrfTc
# *****************************************************************************


# *****************************************************************************
# *** Function for fast calculation of residuals

# aryMdlParamsChnk[model-ID, parameter-value]
# aryPixConv[x-pixels, y-pixels, conditions, volumes]

cdef float[:, :, :] cy_prf_conv(float[:, :] aryX_view,
                                float[:, :] aryY_view,
                                float[:, :] aryMdlParams_view,
                                float[:, :, :, :] aryPixConv_view,
                                float[:, :, :] aryPrfTc_view,
                                float[:, :] aryGauss_view,
                                float[:, :, :] aryTmp_view,
                                unsigned long varNumMdls,
                                unsigned long varNumCon,
                                unsigned long varNumVol,
                                unsigned int varNumX,
                                unsigned int varNumY):

    cdef float varPosX, varPosY, varSd, varPi, varSum, varTmp
    cdef unsigned int idxX, idxY
    cdef unsigned long

    varPi = 3.14159265

    # Loop through conditions and volumes, and call cdef:
    for idxCon in range(varNumCon):

        # Loop through combinations of model parameters:
        for idxMdl in range(varNumMdls):

            # Spatial parameters of current model:
            varPosX = aryMdlParams_view[idxMdl, 1]
            varPosY = aryMdlParams_view[idxMdl, 2]
            varSd = aryMdlParams_view[idxMdl, 3]

            # Create Gaussian pRF model:
            for idxX in range(varNumX):
                for idxY in range(varNumY):

                    # Create Gaussian:
                    varTmp = (
                      (
                       (aryX_view[idxX, idxY] - varPosX) ** 2
                       + (aryY_view[idxX, idxY] - varPosY) ** 2
                       )
                      / (2 * (varSd ** 2))
                      )

                    # Scale Gaussian:
                    aryGauss_view[idxX, idxY] = (
                        c_exp(-varTmp)
                        / (2 * varPi * (varSd ** 2))
                        )

            # Multiply pixel-time courses with Gaussian pRF models:
            for idxVol in range(varNumVol):
                for idxX in range(varNumX):
                    for idxY in range(varNumY):
                        aryTmp_view[idxX, idxY, idxVol] = (
                            aryPixConv_view[idxX, idxY, idxCon, idxVol]
                            * aryGauss_view[idxX, idxY]
                            )

            # Calculate sum across x- and y-dimensions - the 'area under
            # the Gaussian surface'. This gives us the ratio of
            # 'activation' of the pRF at each time point, or, in other
            # words, the pRF time course model. Note: Normalisation of pRFs
            # takes at funcGauss(); pRF models are normalised to have an
            # area under the curve of one when they are created.
            for idxVol in range(varNumVol):

                # Variable for calculation of sum (area under the
                # Gaussian):
                varSum = 0

                # Loop through pixel in current volume:
                for idxX in range(varNumX):
                    for idxY in range(varNumY):
                        varSum += aryTmp_view[idxX, idxY, idxVol]

                aryPrfTc_view[idxMdl, idxCon, idxVol] = varSum

    # Return memory views:
    return aryPrfTc_view
# *****************************************************************************
