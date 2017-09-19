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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


def prf_par(aryMdlParamsChnk, tplVslSpcHighSze, varNumVol, aryPngDataHigh,
                  queOut):
    """
    Create pRF time course models.


    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Output array with pRF model time courses:
    aryOut = np.zeros([varChnkSze, varNumVol])

    # Loop through combinations of model parameters:
    for idxMdl in range(0, varChnkSze):

        # Depending on the relation between the number of x- and y-positions
        # at which to create pRF models and the size of the super-sampled
        # visual space, the indicies need to be rounded:
        varTmpX = np.around(aryMdlParamsChnk[idxMdl, 1], 0)
        varTmpY = np.around(aryMdlParamsChnk[idxMdl, 2], 0)
        varTmpSd = np.around(aryMdlParamsChnk[idxMdl, 3], 0)

        # Create pRF model (2D):
        aryGauss = funcGauss(tplVslSpcHighSze[0],
                             tplVslSpcHighSze[1],
                             varTmpX,
                             varTmpY,
                             varTmpSd)

        # Multiply super-sampled pixel-time courses with Gaussian pRF models:
        aryPrfTcTmp = np.multiply(aryPngDataHigh, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the pRF
        # time course model (i.e. not yet scaled for the size of the pRF).
        aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1))

        # Normalise the pRF time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point, or,
        # in other words, the pRF time course model. REMOVED - normalisation
        # has been moved to funcGauss(); pRF models are normalised when to have
        # an area under the curve of one when they are created.
        # aryPrfTcTmp = np.divide(aryPrfTcTmp,
        #                         np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idxMdl, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    aryOut = np.hstack((np.array(aryMdlParamsChnk[:, 0], ndmin=2).T,
                        aryOut))

    # Put output to queue:
    queOut.put(aryOut)
