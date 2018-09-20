# -*- coding: utf-8 -*-
"""Load PNGs for pRF model creation."""

# Part of py_pRF_mapping library
# Copyright (C) 2018  Ingo Marquardt
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

import numpy as np
from PIL import Image


def load_png(varNumVol, lstPathPng, tplVslSpcSze=(200, 200), varStrtIdx=0,
             varZfill=3):
    """
    Load PNGs with stimulus information for pRF model creation.

    Parameters
    ----------

    varNumVol : int
        Number of PNG files.

    lstPathPng : lst
        Basename of the screenshots (PNG images) of pRF stimuli. List of
        strings with one path per experimental run. PNG files can be created by
        running `~/pyprf/stimulus_presentation/code/stimulus.py` with 'Logging
        mode' set to 'True'. E.g.: `lstPathPng = ['~/stimuli/run_01_frame_',
        '~/stimuli/run_02_frame_']`.
    tplVslSpcSze : tuple
        Pixel size (x, y) at which PNGs are sampled. In case of large PNGs it
        is useful to sample at a lower than the original resolution.
    varStrtIdx : int
        Start index of PNG files. For instance, `varStrtIdx = 0` if the name of
        the first PNG file is `file_000.png`, or `varStrtIdx = 1` if it is
        `file_001.png`.
    varZfill : int
        Zero padding of PNG file names. For instance, `varStrtIdx = 3` if the
        name of PNG files is `file_007.png`, or `varStrtIdx = 4` if it is
        `file_0007.png`.

    Returns
    -------
    aryPngData : np.array
        3D Numpy array with the following structure:
        aryPngData[x-pixel-index, y-pixel-index, PngNumber]

    Notes
    -----
    Part of py_pRF_mapping library.
    """
    # Number of runs:
    varNumRun = len(lstPathPng)

    # Total number of PNGs (i.e. total number of frames in all runs):
    varNumPng = int(varNumVol) * varNumRun

    # Create list with complete path & file names of all of png files to load.
    lstAllPngs = [None] * varNumPng
    varCntPng = 0
    for idxRun in range(varNumRun):
        for idxVol in range(varNumVol):
            lstAllPngs[varCntPng] = (lstPathPng[idxRun]
                                     + str(idxVol + varStrtIdx).zfill(varZfill)
                                     + '.png')
            varCntPng += 1

    # The png data will be saved in a numpy array of the following order:
    # aryPngData[x-pixel, y-pixel, PngNumber].
    aryPngData = np.zeros((tplVslSpcSze[0],
                           tplVslSpcSze[1],
                           varNumPng)).astype(np.uint8)

    # # Open first image in order to check dimensions (greyscale or RGB, i.e. 2D
    # # or 3D).
    # objIm = Image.open(lstAllPngs[0])
    # aryTest = np.array(objIm.resize((objIm.size[0], objIm.size[1]),
    #                                 Image.ANTIALIAS))
    # varNumDim = aryTest.ndim
    # del(aryTest)

    # Loop trough PNG files:
    for idxPng in range(varNumPng):

        # Old version of reading images with scipy
        # aryPngData[:, :, idxVol] = sp.misc.imread(lstAllPngs[idxVol])[:, :, 0]
        # aryPngData[:, :, idxVol] = sp.misc.imread(lstAllPngs[idxVol])[:, :]

        # Load & resize image:
        objIm = Image.open(lstAllPngs[idxPng])

        # Rescale png image to size of visual space model:
        aryTmp = np.array(objIm.resize((tplVslSpcSze[0], tplVslSpcSze[1]),
                          Image.NEAREST))

        # Number of dimensions (two for greyscale image, three for RGB image).
        varNumDim = aryTmp.ndim

        # Casting of array depends on dimensionality (greyscale or RGB, i.e. 2D
        # or 3D).
        # if varNumDim == 2:
        #    # Rescale png image, and put into numpy array:
        #    aryTmp = aryTmp[:, :]
        # elif varNumDim == 3:
        if varNumDim == 3:

            # In case of RGB image, reduce number of dimensions (stimuli are
            # greyscale, so all three RGB values are assumed to be the same).
            aryTmp = aryTmp[:, :, 0]

        # x and y dimension of png image and data array do not match, we
        # turn the image to fit:
        aryTmp = np.rot90(aryTmp, k=3, axes=(0, 1))
        aryPngData[:, :, idxPng] = np.copy(aryTmp)

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 200).astype(np.int8)

    return aryPngData
