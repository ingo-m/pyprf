# -*- coding: utf-8 -*-
"""Create pixel-wise HRF model time courses."""

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
from pRF_utilities import funcHrf, funcConvPar


def funcCrtPixMdl(aryPngData,
                  varNumVol,
                  varTr,
                  tplPngSize,
                  varPar):
    """
    Create pixel-wise HRF model time courses.

    After concatenating all stimulus frames (png files) into an array, this
    stimulus array is effectively a boxcar design matrix with zeros if no
    stimulus was present at that pixel at that frame, and ones if a stimulus
    was present. In this function, we convolve this boxcar design matrix with
    an HRF model.
    """
    print('------Create pixel-wise HRF model time courses')

    # Create 'canonical' HRF time course model:
    vecHrf = funcHrf(varNumVol, varTr)

    # List into which the chunks of input data for the parallel processes will
    # be put:
    lstParData = [None] * varPar

    # Number of pixels:
    varNumPix = tplPngSize[0] * tplPngSize[1]

    # Reshape png data:
    aryPngData = np.reshape(aryPngData,
                            [aryPngData.shape[0] * aryPngData.shape[1],
                             aryPngData[2]]
                            )

    # Vector with the indicies at which the input data will be separated in
    # order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumPix,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumPix))

    # Put input data into chunks:
    for idxChnk in range(0, varPar):
        # Index of first voxel to be included in current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last voxel to be included in current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstParData[idxChnk] = aryPngData[varTmpChnkSrt:varTmpChnkEnd, :]

    # We don't need the original array with the input data anymore:
    # del(aryPngData)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstRes = [None] * varPar

    print('------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcConvPar,
                                     args=(idxPrc,
                                           lstParData[idxPrc],
                                           vecHrf,
                                           varNumVol,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('------Collecting results from parallel processes')

    # Create list for vectors with results from parallel processes, in order to
    # put the results into the correct order:
    lstPixConv = [None] * varPar

    # Put output into correct order:
    for idxRes in range(0, varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstRes[idxRes][0]

        # Put fitting results into list, in correct order:
        lstPixConv[varTmpIdx] = lstRes[idxRes][1]

    # Concatenate convolved pixel time courses (into the same order as they
    # were entered into the analysis):
    aryPixConv = np.zeros(0)
    for idxRes in range(0, varPar):
        aryPixConv = np.append(aryPixConv, lstPixConv[idxRes])

    # Delete unneeded large objects:
    # del(lstRes)
    # del(lstPixConv)

    # Reshape results:
    aryPixConv = np.reshape(aryPixConv,
                            [tplPngSize[0],
                             tplPngSize[1],
                             varNumVol])

    # Return:
    return aryPixConv
