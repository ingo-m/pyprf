# -*- coding: utf-8 -*-
"""Convolve pixel-wise design matrix."""

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
from pyprf.analysis.model_creation_pixelwise_par import conv_par
from pyprf.analysis.utilities import crt_hrf


def conv_dsgn_mat(aryPngData, varTr, varPar=10):
    """
    Convolve pixel-wise design matrix.

    Parameters
    ----------
    aryPngData : np.array
        3D numpy array with the following structure:
        aryPngData[x-pixel-index, y-pixel-index, PngNumber]
    varTr : float
        Volume TR of functional data (needed for convolution of timecourses
        with haemodynamic response function).
    varPar : int
        Number of processes to run in parallel (multiprocessing).

    Returns
    -------
    aryPixConv : np.array
        Numpy array with convolved design matrix, shape:
        aryPixConv[x-pixels, y-pixels, conditions, volumes].

    Notes
    -----
    After concatenating all stimulus frames (png files) into an array, this
    stimulus array is effectively a boxcar design matrix with value `zero` if
    no stimulus was present at that pixel at that frame, and pixel intensity (1
    to 255) if a stimulus was present. Here, this boxcar design matrix is
    convolved with an HRF model.

    The number of GLM predictors is inferred from the number of unique values
    in the image (e.g. greyscale values of 25 and 255 if the stimulus was
    presented at two contrast level).
    """
    # Error message if image is not of expected type:
    strErr = 'Image is not of expected type (not uint8).'

    # Images are expected to have uint8 type.
    assert (type(aryPngData[0, 0, 0]) is np.uint8), strErr

    # Input array is of uint8 datatype (0 to 255). Higher precision is not
    # necessary and not possible, because stimulus screenshots (created in
    # `~/pyprf/pyprf/stimulus_presentation/code/stimulus.py`) are of uint8
    # type. The number of GLM predictors is inferred from the number of unique
    # values in the image. For example, if stimuli were presented at one
    # contrast level only (e.g. maximum contrast, 255) there would be two
    # unique values in the screenshot (0 for rest/background, 255 for
    # stimulus). On the other hand, if there were two different contrast
    # levels, (e.g. greyscale values of 25 and 255), there would be three
    # unique values in the image.
    vecCon = np.unique(aryPngData)
    # NOTE: `np.unique` 'Returns the sorted unique elements of an array.' This
    # is important, because GLM predictors will be assigned in order of
    # ascending contrast.

    # Remove first entry, which should correspond to the rest/background pixel
    # intensity. It should be zero.
    assert (vecCon[0] == 0), 'Rest/background pixel intensity is not zero.'
    vecCon = vecCon[1:]

    # Number of contrast levels.
    varNumCon = vecCon.shape[0]

    # Get number of volumes from input array:
    varNumVol = aryPngData.shape[2]

    # Remember original size of PNGs:
    tplPngSize = (aryPngData.shape[0], aryPngData.shape[1])

    # Create 'canonical' HRF time course model:
    vecHrf = crt_hrf(varNumVol, varTr)

    # List into which the chunks of input data for the parallel processes will
    # be put:
    lstParData = [None] * varPar

    # Number of pixels:
    varNumPix = aryPngData.shape[0] * aryPngData.shape[1]

    # Reshape png data (so that dimension are
    # `aryPngData[(x-pixel-index * y-pixel-index), PngNumber]`):
    aryPngData = np.reshape(aryPngData,
                            ((aryPngData.shape[0] * aryPngData.shape[1]),
                             aryPngData.shape[2]))

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
    del(aryPngData)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstRes = [None] * varPar

    # print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=conv_par,
                                     args=(idxPrc,
                                           lstParData[idxPrc],
                                           vecCon,
                                           vecHrf,
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

    # print('---------Collecting results from parallel processes')

    # Create list for vectors with results from parallel processes, in order to
    # put the results into the correct order:
    lstPixConv = [None] * varPar

    # Put output into correct order:
    for idxRes in range(varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstRes[idxRes][0]

        # Put fitting results into list, in correct order:
        lstPixConv[varTmpIdx] = lstRes[idxRes][1]

    # Concatenate convolved pixel time courses (into the same order as they
    # were entered into the analysis). Shape: aryPixConv[(x*y pixels),
    # conditions, volumes]
    aryPixConv = np.concatenate(lstPixConv, axis=0)

    # Delete unneeded large objects:
    # del(lstRes)
    # del(lstPixConv)

    # Reshape results:
    aryPixConv = np.reshape(aryPixConv,
                            (tplPngSize[0],
                             tplPngSize[1],
                             varNumCon,
                             varNumVol)).astype(np.float32)
    # New shape: aryPixConv[x-pixels, y-pixels, conditions, volumes]

    # Return:
    return aryPixConv
