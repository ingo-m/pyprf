# -*- coding: utf-8 -*-
"""Main function for pRF finding."""

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
from pyprf.analysis.utilities import cls_set_config
from pyprf.analysis.cython_leastsquares import cy_lst_sq


def find_prf_cpu(idxPrc, dicCnfg, vecMdlXpos, vecMdlYpos, vecMdlSd,  #noqa
                 aryFuncChnk, aryPrfTc, strVersion, queOut):
    """
    Find best fitting pRF model for voxel time course, using the CPU.

    Parameters
    ----------
    idxPrc : int
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    dicCnfg : dict
        Dictionary containing config parameters.
    vecMdlXpos : np.array
        1D array with pRF model x positions.
    vecMdlYpos : np.array
        1D array with pRF model y positions.
    vecMdlSd : np.array
        1D array with pRF model sizes (SD of Gaussian).
    aryFunc : np.array
        2D array with functional MRI data, with shape aryFunc[voxel, time].
    aryPrfTc : np.array
        Array with pRF model time courses, with shape
        aryPrfTc[x-pos, y-pos, SD, motion-direction, time]
    strVersion : str
        Which version to use for pRF finding; 'numpy' or 'cython'.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------
    lstOut : list
        List containing the following objects:
        idxPrc : int
            Process ID of the process calling this function (for CPU
            multi-threading). In GPU version, this parameter is 0.
        vecBstXpos : np.array
            1D array with best fitting x-position for each voxel, with shape
            vecBstXpos[voxel].
        vecBstYpos : np.array
            1D array with best fitting y-position for each voxel, with shape
            vecBstYpos[voxel].
        vecBstSd : np.array
            1D array with best fitting pRF size for each voxel, with shape
            vecBstSd[voxel].
        vecBstR2 : np.array
            1D array with R2 value of 'winning' pRF model for each voxel, with
            shape vecBstR2[voxel].

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue. This version performs the model finding on the CPU,
    using numpy or cython (depending on the value of `strVersion`).
    """
    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Conditional imports:
    #if cfg.strVersion == 'cython':
    #    from cython_leastsquares import cy_lst_sq

    # Number of modelled x-positions in the visual space:
    varNumX = aryPrfTc.shape[0]
    # Number of modelled y-positions in the visual space:
    varNumY = aryPrfTc.shape[1]
    # Number of modelled pRF sizes:
    varNumPrfSizes = aryPrfTc.shape[2]

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    varNumVol = aryFuncChnk.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk), 100000000.0).astype(np.float32)

    # Vector that will hold the temporary residuals from the model fitting:
    # vecTmpRes = np.zeros(varNumVoxChnk).astype(np.float32)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Prepare data for cython (i.e. accelerated) least squares finding:
    if strVersion == 'cython':
        # Instead of fitting a constant term, we subtract the mean from the
        # data and from the model ("FSL style"). First, we subtract the mean
        # over time from the data:
        aryFuncChnkTmean = np.array(np.mean(aryFuncChnk, axis=0), ndmin=2)
        aryFuncChnk = np.subtract(aryFuncChnk, aryFuncChnkTmean[0, None])
        # Secondly, we subtract the mean over time form the pRF model time
        # courses. The array has four dimensions, the 4th is time (one to three
        # are x-position, y-position, and pRF size (SD)).
        aryPrfTcTmean = np.mean(aryPrfTc, axis=3)
        aryPrfTc = np.subtract(aryPrfTc, aryPrfTcTmean[:, :, :, None])
    # Otherwise, create constant term for numpy least squares finding:
    elif strVersion == 'numpy':
        # Constant term for the model:
        vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumMdls = (varNumX * varNumY * varNumPrfSizes)

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrf = np.ceil(vecStatPrf)
        vecStatPrf = vecStatPrf.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # There can be pRF model time courses with a variance of zero (i.e. pRF
    # models that are not actually responsive to the stimuli). For time
    # efficiency, and in order to avoid division by zero, we ignore these
    # model time courses.
    aryPrfTcVar = np.var(aryPrfTc, axis=3)

    # Zero with float32 precision for comparison:
    varZero32 = np.array(([0.0])).astype(np.float32)[0]

    # Loop through pRF models:
    for idxX in range(0, varNumX):

        for idxY in range(0, varNumY):

            for idxSd in range(0, varNumPrfSizes):

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Status indicator:
                    if varCntSts02 == vecStatPrf[varCntSts01]:

                        # Prepare status message:
                        strStsMsg = ('------------Progress: ' +
                                     str(vecStatPrc[varCntSts01]) +
                                     ' % --- ' +
                                     str(vecStatPrf[varCntSts01]) +
                                     ' pRF models out of ' +
                                     str(varNumMdls))

                        print(strStsMsg)

                        # Only increment counter if the last value has not been
                        # reached yet:
                        if varCntSts01 < varStsStpSze:
                            varCntSts01 = varCntSts01 + int(1)

                # Only fit pRF model if variance is not zero:
                if np.greater(aryPrfTcVar[idxX, idxY, idxSd], varZero32):

                    # Calculation of the ratio of the explained variance (R
                    # square) for the current model for all voxel time courses.

                    # Cython version:
                    if strVersion == 'cython':

                        # A cython function is used to calculate the residuals
                        # of the current model:
                        vecTmpRes = cy_lst_sq(
                            aryPrfTc[idxX, idxY, idxSd, :].flatten(),
                            aryFuncChnk)

                    # Numpy version:
                    elif strVersion == 'numpy':

                        # Current pRF time course model:
                        vecMdlTc = aryPrfTc[idxX, idxY, idxSd, :].flatten()

                        # We create a design matrix including the current pRF
                        # time course model, and a constant term:
                        aryDsgn = np.vstack([vecMdlTc,
                                             vecConst]).T

                        # Change type to float32:
                        aryDsgn = aryDsgn.astype(np.float32)

                        # Calculate the least-squares solution for all voxels:
                        vecTmpRes = np.linalg.lstsq(aryDsgn,
                                                    aryFuncChnk,
                                                    rcond=None)[1]

                    # Check whether current residuals are lower than previously
                    # calculated ones:
                    vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

                    # Replace best x and y position values, and SD values.
                    vecBstXpos[vecLgcTmpRes] = vecMdlXpos[idxX]
                    vecBstYpos[vecLgcTmpRes] = vecMdlYpos[idxY]
                    vecBstSd[vecLgcTmpRes] = vecMdlSd[idxSd]

                    # Replace best residual values:
                    vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Increment status indicator counter:
                    varCntSts02 = varCntSts02 + 1

    # After finding the best fitting model for each voxel, we still have to
    # calculate the coefficient of determination (R-squared) for each voxel. We
    # start by calculating the total sum of squares (i.e. the deviation of the
    # data from the mean). The mean of each time course:
    vecFuncMean = np.mean(aryFuncChnk, axis=0)
    # Deviation from the mean for each datapoint:
    vecFuncDev = np.subtract(aryFuncChnk, vecFuncMean[None, :])
    # Sum of squares:
    vecSsTot = np.sum(np.power(vecFuncDev,
                               2.0),
                      axis=0)
    # Coefficient of determination:
    vecBstR2 = np.subtract(1.0,
                           np.divide(vecBstRes,
                                     vecSsTot))

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              vecBstR2]

    queOut.put(lstOut)
