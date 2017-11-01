# -*- coding: utf-8 -*-
"""Load PNGs for pRF model creation."""

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
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from PIL import Image


def load_png(varNumVol, strPathPng, tplVslSpcSze=(200, 200), varStrtIdx=0,
             varZfill=3):
    """
    Load PNGs with stimulus information for pRF model creation.

    Parameters
    ----------
    varNumVol : int
        Number of PNG files.
    strPathPng : str
        Parent directory of PNG files. PNG files need to be organsied in
        numerical order (e.g. `file_001.png`, `file_002.png`, etc.).
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
    # Create list of png files to load:
    lstPngPaths = [None] * varNumVol
    for idx01 in range(0, varNumVol):
        lstPngPaths[idx01] = (strPathPng
                              + str(idx01 + varStrtIdx).zfill(varZfill)
                              + '.png')

    # The png data will be saved in a numpy array of the following order:
    # aryPngData[x-pixel, y-pixel, PngNumber].
    aryPngData = np.zeros((tplVslSpcSze[0],
                           tplVslSpcSze[1],
                           varNumVol))

    # Open first image in order to check dimensions (greyscale or RGB, i.e. 2D
    # or 3D).
    objIm = Image.open(lstPngPaths[0])
    aryTest = np.array(objIm.resize((objIm.size[0], objIm.size[1]),
                                    Image.ANTIALIAS))
    varNumDim = aryTest.ndim
    del(aryTest)

    # Loop trough PNG files:
    for idx01 in range(0, varNumVol):

        # Old version of reading images with scipy
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :, 0]
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :]

        # Load & resize image:
        objIm = Image.open(lstPngPaths[idx01])
        objIm = objIm.resize((tplVslSpcSze[0],
                              tplVslSpcSze[1]),
                             resample=Image.NEAREST)

        # Casting of array depends on dimensionality (greyscale or RGB, i.e. 2D
        # or 3D):
        if varNumDim == 2:
            aryPngData[:, :, idx01] = np.array(objIm.resize(
                (objIm.size[0], objIm.size[1]), Image.ANTIALIAS))[:, :]
        elif varNumDim == 3:
            aryPngData[:, :, idx01] = np.array(objIm.resize(
                (objIm.size[0], objIm.size[1]), Image.ANTIALIAS))[:, :, 0]
        else:
            # Error message:
            strErrMsg = ('ERROR: PNG files for model creation need to be RGB '
                         + 'or greyscale.')
            raise ValueError(strErrMsg)

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 200).astype(np.int8)

    return aryPngData
