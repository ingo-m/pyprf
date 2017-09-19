# -*- coding: utf-8 -*-
"""pRF finding function definitions."""

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
import scipy as sp
import nibabel as nb
from scipy.stats import gamma


def load_nii(strPathIn):
    """
    Load nii file.

    Parameters
    ----------
    strPathIn : str
        Path to nifti file to load.

    Returns
    -------
    aryNii : np.array
        Array containing nii data. 32 bit floating point precision.
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.
    """
    # Load nii file (this doesn't load the data into memory yet):
    objNii = nb.load(strPathIn)
    # Load data into array:
    aryNii = np.asarray(objNii.dataobj).astype(np.float32)
    # Get headers:
    objHdr = objNii.header
    # Get 'affine':
    aryAff = objNii.affine
    # Output nii data as numpy array and header:
    return aryNii, objHdr, aryAff


def load_nii_large(strPathIn):
    """
    Load large nii file volume by volume, at float32 precision.

    Parameters
    ----------
    strPathIn : str
        Path to nifti file to load.

    Returns
    -------
    aryNii : np.array
        Array containing nii data. 32 bit floating point precision.
    objHdr : header object
        Header of nii file.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of nii data.
    """
    # Load nii file (this does not load the data into memory yet):
    objNii = nb.load(strPathIn)
    # Get image dimensions:
    tplSze = objNii.shape
    # Create empty array for nii data:
    aryNii = np.zeros(tplSze, dtype=np.float32)

    # Loop through volumes:
    for idxVol in range(tplSze[3]):
        aryNii[..., idxVol] = np.asarray(
              objNii.dataobj[..., idxVol]).astype(np.float32)

    # Get headers:
    objHdr = objNii.header
    # Get 'affine':
    aryAff = objNii.affine
    # Output nii data as numpy array and header:
    return aryNii, objHdr, aryAff


def crt_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """
    Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.

    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX))
         + np.square((aryY - varPosY))
         ) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2.0 * np.pi * np.square(varSd))

    return aryGauss


def crt_hrf(varNumVol, varTr):
    """Create double gamma function.

    Source:
    http://www.jarrodmillman.com/rcsds/lectures/convolution_background.html
    """
    vecX = np.arange(0, varNumVol, 1)

    # Expected time of peak of HRF [s]:
    varHrfPeak = 6.0 / varTr
    # Expected time of undershoot of HRF [s]:
    varHrfUndr = 12.0 / varTr
    # Scaling factor undershoot (relative to peak):
    varSclUndr = 0.35

    # Gamma pdf for the peak
    vecHrfPeak = gamma.pdf(vecX, varHrfPeak)
    # Gamma pdf for the undershoot
    vecHrfUndr = gamma.pdf(vecX, varHrfUndr)
    # Combine them
    vecHrf = vecHrfPeak - varSclUndr * vecHrfUndr

    # Scale maximum of HRF to 1.0:
    vecHrf = np.divide(vecHrf, np.max(vecHrf))

    return vecHrf
