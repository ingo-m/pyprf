# -*- coding: utf-8 -*-
"""Main function for preprocessing of data & models."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
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

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d

# %% Linear trend removal from fMRI data


def funcLnTrRm(aryFunc, varSdSmthSpt):
    """
    Perform linear trend removal on the input fMRI data.

    The variable varSdSmthSpt is not needed, only included for consistency
    with other functions using the same parallelisation.
    """
    # Number of voxels in this chunk:
    # varNumVoxChnk = aryFunc.shape[0]

    # Number of time points in this chunk:
    varNumVol = aryFunc.shape[1]

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFunc = aryFunc.T

    # Linear mode to fit to the voxel time courses:
    vecMdlTc = np.linspace(0,
                           1,
                           num=varNumVol,
                           endpoint=True)
    # vecMdlTc = vecMdlTc.flatten()

    # We create a design matrix including the linear trend and a
    # constant term:
    aryDsgn = np.vstack([vecMdlTc, np.ones(len(vecMdlTc))]).T

    # Calculate the least-squares solution for all voxels:
    aryLstSqFt = np.linalg.lstsq(aryDsgn, aryFunc)[0]

    # Multiply the linear term with the respective parameters to obtain the
    # fitted line for all voxels:
    aryLneFt = np.multiply(vecMdlTc[:, None], aryLstSqFt[0, :])

    # Using the least-square fitted model parameters, we remove the linear
    # term from the data:
    aryFunc = np.subtract(aryFunc,
                          aryLneFt)

    # Using the constant term, we remove the mean from the data:
    # aryFunc = np.subtract(aryFunc,
    #                           aryLstSqFt[1, :])

    # Bring array into original order (time from left to right):
    aryFunc = aryFunc.T

    return aryFunc

# %% Spatial smoothing of fMRI data


def funcSmthSpt(aryFuncChnk, varSdSmthSpt):
    """Apply spatial smoothing to the input data.

    Parameters
    ----------
    aryFuncChnk : np.array
        TODO
    varSdSmthSpt : float (?)
        Extent of smoothing.
    Returns
    -------
    aryFuncChnk : np.array
        Smoothed data.
    """
    varNdim = aryFuncChnk.ndim

    # Number of time points in this chunk:
    varNumVol = aryFuncChnk.shape[-1]

    # Loop through volumes:
    if varNdim == 4:
        for idxVol in range(0, varNumVol):

            aryFuncChnk[:, :, :, idxVol] = gaussian_filter(
                aryFuncChnk[:, :, :, idxVol],
                varSdSmthSpt,
                order=0,
                mode='nearest',
                truncate=4.0)
    elif varNdim == 5:
        varNumMtnDrctns = aryFuncChnk.shape[3]
        for idxVol in range(0, varNumVol):
            for idxMtn in range(0, varNumMtnDrctns):
                aryFuncChnk[:, :, :, idxMtn, idxVol] = gaussian_filter(
                    aryFuncChnk[:, :, :, idxMtn, idxVol],
                    varSdSmthSpt,
                    order=0,
                    mode='nearest',
                    truncate=4.0)

    # Output list:
    return aryFuncChnk

# %% Temporal smoothing of fMRI data & pRF time course models


def funcSmthTmp(aryFuncChnk, varSdSmthTmp):
    """Apply temporal smoothing to fMRI data & pRF time course models.

    Parameters
    ----------
    aryFuncChnk : np.array
        TODO
    varSdSmthTmp : float (?)
        extend of smoothing

    Returns
    -------
    aryFuncChnk : np.array
        TODO
    """
    # For the filtering to perform well at the ends of the time series, we
    # set the method to 'nearest' and place a volume with mean intensity
    # (over time) at the beginning and at the end.
    aryFuncChnkMean = np.mean(aryFuncChnk,
                              axis=1,
                              keepdims=True)

    aryFuncChnk = np.concatenate((aryFuncChnkMean,
                                  aryFuncChnk,
                                  aryFuncChnkMean), axis=1)

    # In the input data, time goes from left to right. Therefore, we apply
    # the filter along axis=1.
    aryFuncChnk = gaussian_filter1d(aryFuncChnk,
                                    varSdSmthTmp,
                                    axis=1,
                                    order=0,
                                    mode='nearest',
                                    truncate=4.0)

    # Remove mean-intensity volumes at the beginning and at the end:
    aryFuncChnk = aryFuncChnk[:, 1:-1]

    # Output list:
    return aryFuncChnk