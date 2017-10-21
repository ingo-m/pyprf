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


def conv_par(idxPrc, aryPngData, vecHrf, queOut):
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
            `aryPixConv[(x-pixel-index * y-pixel-index), PngNumber`. Same shape
            as input (`aryPngData`).

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.

    Notes
    ---
    The pixel-wise design matrix is convolved with an HRF model.
    """
    # Array for function output (convolved pixel-wise time courses):
    aryPixConv = np.zeros(np.shape(aryPngData), dtype=np.float32)

    # Explicity typing. NOTE: input to `np.convolve` function needs to be
    # float64 to avoid errors.
    vecHrf = vecHrf.astype(np.float64)

    # Number of volumes:
    varNumVol = aryPngData.shape[1]

    # Each pixel time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through pixels:
    for idxPix in range(0, aryPngData.shape[0]):

        # Extract the current pixel time course:
        vecDm = aryPngData[idxPix, :].astype(np.float64)

        # In order to avoid an artefact at the end of the time series, we have
        # to concatenate an empty array to both the design matrix and the HRF
        # model before convolution. NOTE: input to `np.convolve` function needs
        # to be float64 to avoid errors.
        vecZeros = np.zeros([100, 1], dtype=np.float64).flatten()
        vecDm = np.concatenate((vecDm, vecZeros))
        vecHrf = np.concatenate((vecHrf, vecZeros))

        # Convolve design matrix with HRF model. NOTE: input to `np.convolve`
        # function needs to be float64 to avoid errors.
        aryPixConv[idxPix, :] = np.convolve(vecDm,
                                            vecHrf,
                                            mode='full'
                                            )[0:varNumVol].astype(np.float32)

    # Create list containing the convolved pixel-wise timecourses, and the
    # process ID:
    lstOut = [idxPrc, aryPixConv]

    # Put output to queue:
    queOut.put(lstOut)
