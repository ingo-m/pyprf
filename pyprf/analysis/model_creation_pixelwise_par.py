# -*- coding: utf-8 -*-
"""Parallelisation function for conv_dsgn_mat."""

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
from scipy.signal import fftconvolve


def conv_par(idxPrc, aryPngData, vecCon, vecHrf, queOut):
    """
    Parallelised convolution of pixel-wise design matrix.

    Parameters
    ----------
    idxPrc : int
        Process index (this function can be called in parallel, and the process
        index can be used to identify which return value belongs to which
        process).
    aryPngData : np.array
        2D numpy array with the following structure:
        `aryPngData[(x-pixel-index * y-pixel-index), PngNumber]`
    vecCon : np.array
        1D numpy array with stimulus contrast values (e.g. [255] if only
        maximum contrast was presented, or [25, 255] two contrast levels were
        presented).
    vecHrf : np.array
        1D numpy array with HRF time course model.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------
    lstOut : list
        List containing the following objects:
        idxPrc : int
            Process ID of the process calling this function (for CPU
            multi-threading). In GPU version, this parameter is 0.
        aryPixConv : np.array
            Numpy array containing convolved design matrix. Dimensionality:
            `aryPixConv[(x*y pixels), conditions, volumes]`.

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.

    Notes
    ---
    The pixel-wise design matrix is convolved with an HRF model.
    """
    # Number of contrast levels.
    varNumCon = vecCon.shape[0]

    # Number of pixels:
    varNumPix = aryPngData.shape[0]

    # Number of volumes:
    varNumVol = aryPngData.shape[1]

    # Array for function output (convolved pixel-wise time courses), shape:
    # aryPixConv[(x*y pixels), conditions, volumes].
    aryPixConv = np.zeros((varNumPix,
                           varNumCon,
                           varNumVol), dtype=np.float32)

    # Binarise stimulus condition information. `aryPngData` contains pixel
    # intensity in greyscale intensity values (between 0 and 255). We binarise
    # this information by adding another dimension for stimulus level. Thus,
    # whereas originally there is one timecourse for each pixel (containing n
    # contrast levels, e.g. 0, 25, and 255), in the new array there are n
    # timecourses per pixel, each containign only 0 and 1.
    aryPngCon = np.zeros((varNumPix, varNumCon, varNumVol), dtype=np.bool)

    # aryPngData = aryPngData.astype(np.uint8)
    # vecCon = vecCon.astype(np.uint8)

    # Loop through conditions:
    for idxCon in range(varNumCon):
        aryPngCon[:, idxCon, :] = np.equal(aryPngData, vecCon[idxCon])

    # Explicity typing. NOTE: input to `np.convolve` function needs to be
    # float64 to avoid errors.
    vecHrf = vecHrf.astype(np.float64)

    # In order to avoid an artefact at the end of the time series, we have to
    # concatenate an empty array to both the design matrix and the HRF model
    # before convolution.
    vecZeros = np.zeros([100, 1], dtype=np.float64).flatten()
    vecHrf = np.concatenate((vecHrf, vecZeros))

    # Each pixel time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through conditions & pixels.
    for idxCon in range(varNumCon):
        for idxPix in range(varNumPix):

            # Extract the current pixel time course. NOTE: input to
            # `np.convolve` function needs to be float64 to avoid errors.
            vecDm = aryPngCon[idxPix, idxCon, :].astype(np.float64)

            # In order to avoid an artefact at the end of the time series, we
            # have to concatenate an empty array to both the design matrix and
            # the HRF model before convolution.
            vecDm = np.concatenate((vecDm, vecZeros))

            # Convolve design matrix with HRF model.
            # aryPixConv[idxPix, idxCon, :] = np.convolve(
            #     vecDm, vecHrf, mode='full')[0:varNumVol].astype(np.float32)
            aryPixConv[idxPix, idxCon, :] = fftconvolve(
                vecDm, vecHrf, mode='full')[0:varNumVol].astype(np.float32)

    # Create list containing the convolved pixel-wise timecourses, and the
    # process ID:
    lstOut = [idxPrc, aryPixConv]

    # Put output to queue:
    queOut.put(lstOut)
