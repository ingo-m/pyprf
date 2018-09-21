# -*- coding: utf-8 -*-
"""Create pRF time courses models."""

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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import multiprocessing as mp
from pyprf.analysis.model_creation_timecourses_par import prf_par


def crt_prf_tcmdl(aryPixConv, tplVslSpcSze=(200, 200), varNumX=40, varNumY=40,  #noqa
                  varExtXmin=-5.19, varExtXmax=5.19, varExtYmin=-5.19,
                  varExtYmax=5.19, varPrfStdMin=0.1, varPrfStdMax=7.0,
                  varNumPrfSizes=40, varPar=10):
    """
    Create pRF time courses models.

    Parameters
    ----------
    aryPixConv : np.array
        4D numpy array containing the pixel-wise, HRF-convolved design matrix,
        with the following structure: `aryPixConv[aryPixConv[x-pixels,
        y-pixels, conditions, volumes]`.
    tplVslSpcSze : tuple
        Pixel size of visual space model in which the pRF models are created
        (x- and y-dimension).
    varNumX : int
        Number of x-positions in the visual space to model.
    varNumY : int
        Number of y-positions in the visual space to model.
    varExtXmin : float
        Extent of visual space from centre of the screen in negative
        x-direction (i.e. from the fixation point to the left end of the
        screen) in degrees of visual angle.
    varExtXmax : float
        Extent of visual space from centre of the screen in positive
        x-direction (i.e. from the fixation point to the right end of the
        screen) in degrees of visual angle.
    varExtYmin : float
        Extent of visual space from centre of the screen in negative
        y-direction (i.e. from the fixation point to the lower end of the
        screen) in degrees of visual angle.
    varExtYmax : float
        Extent of visual space from centre of the screen in positive
        y-direction (i.e. from the fixation point to the upper end of the
        screen) in degrees of visual angle.
    varPrfStdMin : flaot
        Minimum pRF model size (standard deviation of 2D Gaussian) in  degrees
        of visual angle.
    varPrfStdMax : flaot
        Maximum pRF model size (standard deviation of 2D Gaussian) in  degrees
        of visual angle.
    varNumPrfSizes : int
        Number of pRF sizes to model.
    varPar : int
        Number of processes to run in parallel (multiprocessing).

    Returns
    -------
    aryPrfTc5D : np.array
        5D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc5D[x-position, y-position, SD, condition, volume]`.

    Notes
    -----
    This function creates the pRF time course models, from which the best-
    fitting model for each voxel will be selected.
    """
    # Number of conditions:
    varNumCon = aryPixConv.shape[2]

    # Number of volumes:
    varNumVol = aryPixConv.shape[3]

    # Only fit pRF models if dimensions of pRF time course models are
    # correct.
    strErrMsg = ('Aspect ratio of visual space models does not agree with'
                 + ' specified number of pRFs to model.')
    lgcAssert = ((float(tplVslSpcSze[0]) / float(varNumX))
                 == (float(tplVslSpcSze[1]) / float(varNumY)))
    assert lgcAssert, strErrMsg

    # Calculate the scaling factor from degrees of visual angle to pixels in
    # the upsampled visual space separately for the x- and the y-directions
    # (the two should be the same).
    varDgr2PixUpX = float(tplVslSpcSze[0]) / float(varExtXmax - varExtXmin)
    varDgr2PixUpY = float(tplVslSpcSze[1]) / float(varExtYmax - varExtYmin)

    # The factor relating pixels in the upsampled visual space to degrees of
    # visual angle should be roughly the same (allowing for some rounding error
    # if the visual stimulus was not square).
    strErrMsg = ('The ratio of X and Y dimensions in stimulus space (in '
                 + 'degrees of visual angle) and the ratio of X and Y '
                 + 'dimensions in the upsampled visual space do not agree.')
    lgcAssert = (np.absolute((varDgr2PixUpX - varDgr2PixUpY)) < 0.5)
    assert lgcAssert, strErrMsg

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0,
                       (tplVslSpcSze[0] - 1),
                       varNumX,
                       endpoint=True,
                       dtype=np.float32)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0,
                       (tplVslSpcSze[1] - 1),
                       varNumY,
                       endpoint=True,
                       dtype=np.float32)

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin,
                           varPrfStdMax,
                           varNumPrfSizes,
                           endpoint=True,
                           dtype=np.float32)

    # We multiply the vector with the pRF sizes to be modelled with the scaling
    # factor (for the x-dimensions - as we have just found out, the scaling
    # factors for the x- and y-direction are identical, except for rounding
    # error). Now the vector with the pRF sizes to be modelled is can directly
    # be used for the creation of Gaussian pRF models in upsampled visual
    # space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixUpX, dtype=np.float32)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4), dtype=np.float32)

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = float(varCntMdlPrms)
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = [None] * varPar

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumMdls,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumMdls))

    # Put model parameters into chunks:
    for idxChnk in range(0, varPar):
        # Index of first combination of model parameters to be included in
        # current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last combination of model parameters to be included in
        # current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstMdlParams[idxChnk] = aryMdlParams[varTmpChnkSrt:varTmpChnkEnd, :]

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstOut = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Make sure datatype of pixeltimecourses is float32:
    aryPixConv = aryPixConv.astype(np.float32)

    # Create processes:
    for idxPrc in range(varPar):
        lstPrcs[idxPrc] = mp.Process(target=prf_par,
                                     args=(lstMdlParams[idxPrc],
                                           tplVslSpcSze,
                                           aryPixConv,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(varPar):
        lstOut[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(varPar):
        lstPrcs[idxPrc].join()

    # lstOut:
    #        vecMdlIdx : np.array
    #            1D numpy array with model indices (for sorting of models after
    #            parallel function. Shape: vecMdlIdx[varNumMdls].
    #        aryPrfTc : np.array
    #            3D numpy array with pRF model time courses, shape:
    #            aryPrfTc[varNumMdls, varNumCon, varNumVol].

    # Combine model time courses from parallel processes.
    lstMdlIdx = [None] * varPar
    lstPrfTc = [None] * varPar

    # Get vectors with model indicies (vecMdlIdx) and pRF model time courses
    #  from parallel output list.
    for idxPrc in range(varPar):
        lstMdlIdx[idxPrc] = lstOut[idxPrc][0]
        lstPrfTc[idxPrc] = lstOut[idxPrc][1]

    # List to array, concatenating along model-index-dimension:
    vecMdlIdx = np.concatenate(lstMdlIdx, axis=0)
    aryPrfTc = np.concatenate(lstPrfTc, axis=0)

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)
    del(lstMdlIdx)

    # Sort output along the first column (which contains the indicies), so that
    # the output is in the same order as the list of combination of model
    # parameters which we created before the parallelisation:
    aryPrfTc = aryPrfTc[np.argsort(vecMdlIdx, axis=0), :, :]

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNumCon, varNumVol], which
    # will hold the pRF model time courses.
    aryPrfTc5D = np.zeros([varNumX,
                           varNumY,
                           varNumPrfSizes,
                           varNumCon,
                           varNumVol],
                          dtype=np.float32)

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(varNumX):

        # Loop through y-positions:
        for idxY in range(varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 5D array:
                aryPrfTc5D[idxX, idxY, idxSd, :, :] = \
                    aryPrfTc[varCntMdlPrms, :, :]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # Return
    return aryPrfTc5D
