# -*- coding: utf-8 -*-
"""Parallelisation function for pre_pro_func & pre_pro_models."""

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
import time
import multiprocessing as mp
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d


def pre_pro_par(aryFunc, aryMask=np.array([], dtype=np.int16),  #noqa
                lgcLinTrnd=False, varSdSmthTmp=0.0, varSdSmthSpt=0.0,
                varPar=1):
    """
    Preprocess fMRI data or pRF time course models for a pRF analysis.

    Parameters
    ----------
    aryFunc : np.array
        Array with fMRI data or pRF time course models (four-dimensional).
    aryMask : np.array
        If temporal smoothing is supposed to be restricted to a subset of
        voxels, a mask can be provided (e.g. brain mask). 3D, with same spatial
        dimensions as functional dara (aryFunc).
    lgcLinTrnd : bool
        Whether to perform linear trend removal.
    varSdSmthTmp : float
        Extent of temporal smoothing (FWHM) in units of input data (number of
        volumes). No temporal smoothing is applied if varSdSmthTmp = 0.0.
    varSdSmthSpt : float
        Extent of spatial smoothing (FWHM) in units of input data (number of
        voxels). No spatial smoothing is applied if varSdSmthSpt = 0.0.
    varPar : int
        Number of processes to run in parallel.

    Returns
    -------
    aryFunc : np.array
        Preprocessed data, same dimensions as respective input array.

    Notes
    -----
    Spatial smoothing can be applied to fMRI data, and temporal smoothing can
    be applied to both fMRI data or pRF model time courses. Linear trend
    removal can be performed on fMRI data.
    """
    # *************************************************************************
    # ***  Check time
    varTme01 = time.time()
    # *************************************************************************

    # *************************************************************************
    # *** Generic function for parallelisation over voxel time courses

    def funcParVox(funcIn, aryData, aryMask, varSdSmthTmp, varPar):
        """
        Parallelize over another function.

        Data is chunked into arrays of one-dimensional voxel time courses.
        """
        # Shape of input data:
        vecInShp = aryData.shape

        # Number of volumes:
        varNumVol = vecInShp[3]

        # Empty list for results:
        lstResPar = [None] * varPar

        # Empty list for processes:
        lstPrcs = [None] * varPar

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Total number of elements to loop over (voxels):
        varNumEleTlt = (vecInShp[0] * vecInShp[1] * vecInShp[2])

        # Reshape data:
        aryData = np.reshape(aryData, [varNumEleTlt, varNumVol])

        # The exclusion of voxels based on the mask is only used for the fMRI
        # data, not for the pRF time course models. For the pRF time course
        # models, an empty array is passed into this function instead of an
        # actual mask.
        if 0 < aryMask.size:

            # Reshape mask:
            aryMask = np.reshape(aryMask, varNumEleTlt)

            # Take mean over time:
            # aryDataMean = np.mean(aryData, axis=1)

            # Logical test for voxel inclusion: is the voxel value greater than
            # zero in the mask, and is the mean of the functional time series
            # above the cutoff value?
            aryLgc = np.greater(aryMask, 0)

            # Array with functional data for which conditions (mask inclusion
            # and cutoff value) are fullfilled:
            aryData = aryData[aryLgc, :]

        # Number of elements on which function will be applied:
        varNumEleInc = aryData.shape[0]

        print('------------Number of voxels/pRF time courses on which ' +
              'function will be applied: ' + str(varNumEleInc))

        # List into which the chunks of data for the parallel processes will be
        # put:
        lstFunc = [None] * varPar

        # Vector with the indicies at which the data will be separated in order
        # to be chunked up for the parallel processes:
        vecIdxChnks = np.linspace(0,
                                  varNumEleInc,
                                  num=varPar,
                                  endpoint=False)
        vecIdxChnks = np.hstack((vecIdxChnks, varNumEleInc))

        # Put data into chunks:
        for idxChnk in range(0, varPar):
            # Index of first element to be included in current chunk:
            varTmpChnkSrt = int(vecIdxChnks[idxChnk])
            # Index of last element to be included in current chunk:
            varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
            # Put array chunk into list:
            lstFunc[idxChnk] = aryData[varTmpChnkSrt:varTmpChnkEnd, :]

        # We don't need the original array with the functional data anymore:
        del(aryData)

        print('------------Creating parallel processes')

        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=funcIn,
                                         args=(idxPrc,
                                               lstFunc[idxPrc],
                                               varSdSmthTmp,
                                               queOut))
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Start processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(0, varPar):
            lstResPar[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].join()

        print('------------Post-process data from parallel function')

        # Create list for vectors with results, in order to put the results
        # into the correct order:
        lstRes = [None] * varPar

        # Put output into correct order:
        for idxRes in range(0, varPar):

            # Index of results (first item in output list):
            varTmpIdx = lstResPar[idxRes][0]

            # Put results into list, in correct order:
            lstRes[varTmpIdx] = lstResPar[idxRes][1]

        # Merge output vectors (into the same order with which they were put
        # into this function):
        aryRes = np.array([], dtype=np.float32).reshape(0, varNumVol)
        for idxRes in range(0, varPar):
            aryRes = np.append(aryRes, lstRes[idxRes], axis=0)

        # Delete unneeded large objects:
        del(lstRes)
        del(lstResPar)

        # Array for output, same size as input (i.e. accounting for those
        # elements that were masked out):
        aryOut = np.zeros((varNumEleTlt,
                           vecInShp[3]),
                          dtype=np.float32)

        if 0 < aryMask.size:

            # Put results form pRF finding into array (they originally needed
            # to be saved in a list due to parallelisation). If mask was used,
            # we have to account for leaving out some voxels earlier.
            aryOut[aryLgc, :] = aryRes

        else:

            # If no mask was used (for pRF time course models), we don't need
            # to account for leaft out values.
            aryOut = aryRes

        # Reshape pRF finding results:
        aryOut = np.reshape(aryOut,
                            [vecInShp[0],
                             vecInShp[1],
                             vecInShp[2],
                             vecInShp[3]])

        # And... done.
        return aryOut
    # *************************************************************************

    # *************************************************************************
    # *** Generic function for parallelisation over volumes

    # def funcParVol(funcIn, aryData, varSdSmthSpt, varPar):
    #     """
    #     Parallelize over another function.

    #     Data is chunked into separate volumes.
    #     """
    #     # Shape of input data:
    #     vecInShp = aryData.shape

    #     # Number of volumes:
    #     varNumVol = vecInShp[3]

    #     # Empty list for results:
    #     lstResPar = [None] * varPar

    #     # Empty list for processes:
    #     lstPrcs = [None] * varPar

    #     # Create a queue to put the results in:
    #     queOut = mp.Queue()

    #     # List into which the chunks of data for the parallel processes will
    #     # be put:
    #     lstFunc = [None] * varPar

    #     # Vector with the indicies at which the data will be separated in
    #     # order to be chunked up for the parallel processes:
    #     vecIdxChnks = np.linspace(0,
    #                               varNumVol,
    #                               num=varPar,
    #                               endpoint=False)
    #     vecIdxChnks = np.hstack((vecIdxChnks, varNumVol))

    #     # Put data into chunks:
    #     for idxChnk in range(0, varPar):
    #         # Index of first element to be included in current chunk:
    #         varTmpChnkSrt = int(vecIdxChnks[idxChnk])
    #         # Index of last element to be included in current chunk:
    #         varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
    #         # Put array chunk into list:
    #         lstFunc[idxChnk] = aryData[:, :, :, varTmpChnkSrt:varTmpChnkEnd]

    #     # We don't need the original array with the functional data anymore:
    #     del(aryData)

    #     print('------------Creating parallel processes')

    #     # Create processes:
    #     for idxPrc in range(0, varPar):
    #         lstPrcs[idxPrc] = mp.Process(target=funcIn,
    #                                      args=(idxPrc,
    #                                            lstFunc[idxPrc],
    #                                            varSdSmthSpt,
    #                                            queOut))
    #         # Daemon (kills processes when exiting):
    #         lstPrcs[idxPrc].Daemon = True

    #     # Start processes:
    #     for idxPrc in range(0, varPar):
    #         lstPrcs[idxPrc].start()

    #     # Collect results from queue:
    #     for idxPrc in range(0, varPar):
    #         lstResPar[idxPrc] = queOut.get(True)

    #     # Join processes:
    #     for idxPrc in range(0, varPar):
    #         lstPrcs[idxPrc].join()

    #     print('------------Post-process data from parallel function')

    #     # Create list for vectors with results, in order to put the results
    #     # into the correct order:
    #     lstRes = [None] * varPar

    #     # Put output into correct order:
    #     for idxRes in range(0, varPar):

    #         # Index of results (first item in output list):
    #         varTmpIdx = lstResPar[idxRes][0]

    #         # Put results into list, in correct order:
    #         lstRes[varTmpIdx] = lstResPar[idxRes][1]

    #     # Merge output vectors (into the same order with which they were put
    #     # into this function):
    #     aryRes = np.array([], dtype=np.float32).reshape(vecInShp[0],
    #                                                     vecInShp[1],
    #                                                     vecInShp[2],
    #                                                     0)
    #     for idxRes in range(0, varPar):
    #         aryRes = np.append(aryRes, lstRes[idxRes], axis=3)

    #     # Delete unneeded large objects:
    #     del(lstRes)
    #     del(lstResPar)

    #     # And... done.
    #     return aryRes
    # *************************************************************************

    # *************************************************************************
    # *** Linear trend removal for fMRI data

    def funcLnTrRm(idxPrc, aryFuncChnk, varSdSmthSpt, queOut):
        """
        Perform linear trend removal on the input fMRI data.

        The variable varSdSmthSpt is not needed, only included for consistency
        with other functions using the same parallelisation.
        """
        # Number of voxels in this chunk:
        # varNumVoxChnk = aryFuncChnk.shape[0]

        # Number of time points in this chunk:
        varNumVol = aryFuncChnk.shape[1]

        # We reshape the voxel time courses, so that time goes down the column,
        # i.e. from top to bottom.
        aryFuncChnk = aryFuncChnk.T

        # Linear mode to fit to the voxel time courses:
        vecMdlTc = np.linspace(0,
                               1,
                               num=varNumVol,
                               endpoint=True,
                               dtype=np.float32)
        # vecMdlTc = vecMdlTc.flatten()

        # We create a design matrix including the linear trend and a
        # constant term:
        aryDsgn = np.vstack([vecMdlTc,
                             np.ones(len(vecMdlTc), dtype=np.float32)]).T
        aryDsgn = aryDsgn.astype(np.float32, copy=False)

        # Calculate the least-squares solution for all voxels:
        aryLstSqFt = np.linalg.lstsq(aryDsgn, aryFuncChnk, rcond=None)[0]

        # Multiply the linear term with the respective parameters to obtain the
        # fitted line for all voxels:
        aryLneFt = np.multiply(vecMdlTc[:, None],
                               aryLstSqFt[0, :],
                               dtype=np.float32)

        # Using the least-square fitted model parameters, we remove the linear
        # term from the data:
        aryFuncChnk = np.subtract(aryFuncChnk,
                                  aryLneFt,
                                  dtype=np.float32)

        # Using the constant term, we remove the mean from the data:
        # aryFuncChnk = np.subtract(aryFuncChnk,
        #                           aryLstSqFt[1, :])

        # Bring array into original order (time from left to right):
        aryFuncChnk = aryFuncChnk.T

        # Output list:
        lstOut = [idxPrc,
                  aryFuncChnk.astype(np.float32, copy=False)]

        queOut.put(lstOut)
    # *************************************************************************

    # *************************************************************************
    # ***  Spatial smoothing of fMRI data

    # NOTE: This function is not used; because of memory limitations the
    # spatial smoothing should not be parallelised over volumes. The spatial
    # smoothing is performed below without a parallelisation wrapper, using a
    # direct call to the respective numpy function.

    # def funcSmthSpt(idxPrc, aryFuncChnk, varSdSmthSpt, queOut):
    #     """
    #     Apply spatial smoothing to the input data.

    #     The extent of smoothing needs to be specified as an input parameter.
    #     """
    #     # Number of time points in this chunk:
    #     varNumVol = aryFuncChnk.shape[3]

    #     # Input data should already be float32. Just to be sure:
    #     aryFuncChnk = aryFuncChnk.astype(np.float32, copy=False)

    #     # Loop through volumes:
    #     for idxVol in range(0, varNumVol):

    #         aryFuncChnk[:, :, :, idxVol] = gaussian_filter(
    #             aryFuncChnk[:, :, :, idxVol],
    #             varSdSmthSpt,
    #             order=0,
    #             mode='nearest',
    #             truncate=4.0).astype(np.float32, copy=False)

    #     # Output list:
    #     lstOut = [idxPrc,
    #               aryFuncChnk]

    #     queOut.put(lstOut)
    # *************************************************************************

    # *************************************************************************
    # *** Temporal smoothing of fMRI data & pRF time course models

    def funcSmthTmp(idxPrc, aryFuncChnk, varSdSmthTmp, queOut):
        """
        Apply temporal smoothing to the input data.

        The extend of smoothing needs to be specified as an input parameter.
        """
        # For the filtering to perform well at the ends of the time series, we
        # set the method to 'nearest' and place a volume with mean intensity
        # (over time) at the beginning and at the end.
        aryFuncChnkMean = np.mean(aryFuncChnk,
                                  axis=1,
                                  keepdims=True,
                                  dtype=np.float32)

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
        lstOut = [idxPrc,
                  aryFuncChnk.astype(np.float32, copy=False)]

        queOut.put(lstOut)
    # *************************************************************************

    # *************************************************************************
    # *** Apply functions:

    # Data type for all computations should be float32 to avoid memory issues.
    aryFunc = aryFunc.astype(np.float32, copy=False)

    if lgcLinTrnd:
        # Perform linear trend removal (parallelised over voxels):
        print('---------Linear trend removal')
        aryFunc = funcParVox(funcLnTrRm,
                             aryFunc,
                             aryMask,
                             0,
                             varPar)

    # Perform spatial smoothing on fMRI data (reduced parallelisation over
    # volumes because this function is very memory intense):
    if 0.0 < varSdSmthSpt:
        print('---------Spatial smoothing')

        # (A) Reduced parallelisation:

        # varParRed = int(varPar * 0.2)
        # Minimum is one:
        # if varParRed < 1:
        #     varParRed = 1
        # print('------------Reduced parallelisation, number of threads: '
        #      + str(varParRed))

        # aryFunc = funcParVol(funcSmthSpt,
        #                      aryFunc,
        #                      varSdSmthSpt,
        #                      varParRed)

        # (B) No parallelisation:

        # Number of time points in this chunk:
        varNumVol = aryFunc.shape[3]

        # Loop through volumes:
        for idxVol in range(0, varNumVol):

            # print('------------Volume: ' + str(idxVol))

            aryFunc[:, :, :, idxVol] = gaussian_filter(
                aryFunc[:, :, :, idxVol],
                varSdSmthSpt,
                order=0,
                mode='nearest',
                truncate=4.0)

    # Perform temporal smoothing:
    if 0.0 < varSdSmthTmp:
        print('---------Temporal smoothing')
        aryFunc = funcParVox(funcSmthTmp,
                             aryFunc,
                             aryMask,
                             varSdSmthTmp,
                             varPar)
    # *************************************************************************

    # *************************************************************************
    # *** Report time

    varTme02 = time.time()
    varTme03 = varTme02 - varTme01
    print('------Elapsed time: ' + str(varTme03) + ' s')
    print('------Done.')
    # *************************************************************************

    # *************************************************************************
    # Return preprocessed data:
    return aryFunc.astype(np.float32, copy=False)
    # **************************************************************************
