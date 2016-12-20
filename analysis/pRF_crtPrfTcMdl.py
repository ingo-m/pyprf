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
from pRF_utilities import funcPrfTc


def funcCrtPrfTcMdl(tplVslSpcHighSze,  #noqa
                    varNumX,
                    varNumY,
                    varExtXmin,
                    varExtXmax,
                    varExtYmin,
                    varExtYmax,
                    varPrfStdMin,
                    varPrfStdMax,
                    varNumPrfSizes,
                    varPar,
                    varNumVol,
                    aryPngDataHigh):
    """
    Create pRF time courses models.

    ...
    """
    print('------Create pRF time course models')

    # Upsampling factor:
    if (tplVslSpcHighSze[0] / varNumX) == (tplVslSpcHighSze[1] / varNumY):
        varFctUp = tplVslSpcHighSze[0] / varNumX  #noqa
    else:
        print('------ERROR. Dimensions of upsampled visual space do not ' +
              'agree with specified number of pRFs to model.')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0,
                       (tplVslSpcHighSze[0] - 1),
                       varNumX,
                       endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0,
                       (tplVslSpcHighSze[1] - 1),
                       varNumY,
                       endpoint=True)

    # Vector with the standard deviations of the pRF models. We need to convert
    # the standard deviation values from degree of visual angle to the
    # dimensions of the visual space. We calculate the scaling factor from
    # degrees of visual angle to pixels in the *upsampled* visual space
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixUpX = tplVslSpcHighSze[0] / (varExtXmax - varExtXmin)
    varDgr2PixUpY = tplVslSpcHighSze[1] / (varExtYmax - varExtYmin)

    # The factor relating pixels in the upsampled visual space to degrees of
    # visual angle should be roughly the same (allowing for some rounding error
    # if the visual stimulus was not square):
    if 0.5 < np.absolute((varDgr2PixUpX - varDgr2PixUpY)):
        print('------ERROR. The ratio of X and Y dimensions in stimulus ' +
              'space (in degrees of visual angle) and the ratio of X and Y ' +
              'dimensions in the upsampled visual space do not agree')

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin,
                           varPrfStdMax,
                           varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector with the pRF sizes to be modelled with the scaling
    # factor (for the x-dimensions - as we have just found out, the scaling
    # factors for the x- and y-direction are identical, except for rounding
    # error). Now the vector with the pRF sizes to be modelled is can directly
    # be used for the creation of Gaussian pRF models in upsampled visual
    # space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixUpX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

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
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
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
    lstPrfTc = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcPrfTc,
                                     args=(lstMdlParams[idxPrc],
                                           tplVslSpcHighSze,
                                           varNumVol,
                                           aryPngDataHigh,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    # Put output arrays from parallel process into one big array (where each
    # row corresponds to one model time course, the first column corresponds to
    # the index number of the model time course, and the remaining columns
    # correspond to time points):
    aryPrfTc = np.vstack(lstPrfTc)

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Sort output along the first column (which contains the indicies), so that
    # the output is in the same order as the list of combination of model
    # parameters which we created before the parallelisation:
    aryPrfTc = aryPrfTc[np.argsort(aryPrfTc[:, 0])]

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryPrfTc4D = np.zeros([varNumX,
                           varNumY,
                           varNumPrfSizes,
                           varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryPrfTc4D[idxX, idxY, idxSd, :] = aryPrfTc[varCntMdlPrms, 1:]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # Return
    return aryPrfTc4D
