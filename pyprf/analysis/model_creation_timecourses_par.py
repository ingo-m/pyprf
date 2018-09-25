# -*- coding: utf-8 -*-
"""Parallelisation function for crt_prf_tcmdl."""

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

from pyprf.analysis.cython_leastsquares import cy_lst_sq
# from pyprf.analysis.cython_leastsquares_two import cy_lst_sq_two
# from pyprf.analysis.cython_prf_convolve import prf_conv
# import scipy as sp

from pyprf.analysis.utilities import crt_gauss


def prf_par(aryMdlParamsChnk, tplVslSpcSze, aryPixConv, queOut):
    """
    Create pRF time course models.

    Parameters
    ----------
    aryMdlParamsChnk : np.array
        2D numpy array containing the parameters for the pRF models to be
        created. Dimensionality: `aryMdlParamsChnk[model-ID, parameter-value]`.
        For each model there are four values: (0) an index starting from zero,
        (1) the x-position, (2) the y-position, and (3) the standard deviation.
        Parameters 1, 2 , and 3 are in units of the upsampled visual space.
    tplVslSpcSze : tuple
        Pixel size of visual space model in which the pRF models are created
        (x- and y-dimension).
    aryPixConv : np.array
        4D numpy array containing the pixel-wise, HRF-convolved design matrix,
        with the following structure: `aryPixConv[x-pixels, y-pixels,
        conditions, volumes]`.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------
    lstOut : list
        List containing the following object:
        vecMdlIdx : np.array
            1D numpy array with model indices (for sorting of models after
            parallel function. Shape: vecMdlIdx[varNumMdls].
        aryPrfTc : np.array
            3D numpy array with pRF model time courses, shape:
            aryPrfTc[varNumMdls, varNumCon, varNumVol].

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.
    """
    # Number of models (i.e., number of combinations of model parameters in the
    # parallel processing current chunk):
    varNumMdls = aryMdlParamsChnk.shape[0]

    # Number of conditions:
    varNumCon = aryPixConv.shape[2]

    # Number of volumes:
    varNumVol = aryPixConv.shape[3]

    # Number of combinations of model parameters in the current chunk:
    # varNumMdls = np.size(aryMdlParamsChnk, axis=0)

    # Output array with pRF model time courses:
    aryPrfTc = np.zeros([varNumMdls, varNumCon, varNumVol], dtype=np.float32)

    # Loop through combinations of model parameters:
    for idxMdl in range(varNumMdls):

        # Spatial parameters of current model:
        varTmpX = aryMdlParamsChnk[idxMdl, 1]
        varTmpY = aryMdlParamsChnk[idxMdl, 2]
        varTmpSd = aryMdlParamsChnk[idxMdl, 3]

        # Create pRF model (2D):
        aryGauss = crt_gauss(tplVslSpcSze[0],
                             tplVslSpcSze[1],
                             varTmpX,
                             varTmpY,
                             varTmpSd)

        # Multiply super-sampled pixel-time courses with Gaussian pRF models:
        aryPrfTcTmp = np.multiply(aryPixConv, aryGauss[:, :, None, None])
        # Shape: aryPrfTcTmp[x-pixels, y-pixels, conditions, volumes]

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This gives us the ratio of 'activation' of the pRF
        # at each time point, or, in other words, the pRF time course model.
        # Note: Normalisation of pRFs takes at funcGauss(); pRF models are
        # normalised to have an area under the curve of one when they are
        # created.
        aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1), dtype=np.float32)
        # New shape: aryPrfTcTmp[conditions, volumes]

        # Put model time courses into the function's output array:
        aryPrfTc[idxMdl, :, :] = np.copy(aryPrfTcTmp)

    ## Meshgrid for creation of Gaussian pRF models:
    #aryX, aryY = sp.mgrid[0:tplVslSpcSze[0],
    #                      0:tplVslSpcSze[1]]
    #aryX = aryX.astype(np.float32)
    #aryY = aryY.astype(np.float32)

    #aryPrfTc = prf_conv(
    #    aryX,
    #    aryY,
    #    aryMdlParamsChnk,
    #    aryPixConv)

    # Put column with the indicies of model-parameter-combinations into the
    # output list (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    vecMdlIdx = aryMdlParamsChnk[:, 0]

    # Output list:
    lstOut = [vecMdlIdx, aryPrfTc]

    # Put output to queue:
    queOut.put(lstOut)
