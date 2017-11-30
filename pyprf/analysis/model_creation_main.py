# -*- coding: utf-8 -*-
"""pRF model creation."""

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
import nibabel as nb
from pyprf.analysis.model_creation_load_png import load_png
from pyprf.analysis.model_creation_pixelwise import conv_dsgn_mat
from pyprf.analysis.model_creation_timecourses import crt_prf_tcmdl
from pyprf.analysis.utilities import cls_set_config


def model_creation(dicCnfg):
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing config parameters.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    """
    # *************************************************************************
    # *** Load parameters from config file

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)
    # *************************************************************************

    if cfg.lgcCrteMdl:  #noqa

        # *********************************************************************
        # *** Load stimulus information from PNG files:

        print('------Load stimulus information from PNG files')

        aryPngData = load_png(cfg.varNumVol,
                              cfg.strPathPng,
                              cfg.tplVslSpcSze,
                              varStrtIdx=cfg.varStrtIdx,
                              varZfill=cfg.varZfill)
        # *********************************************************************

        # *********************************************************************
        # *** Convolve pixel-wise design matrix with HRF model

        print('------Convolve pixel-wise design matrix with HRF model')

        aryPixConv = conv_dsgn_mat(aryPngData,
                                   cfg.varTr,
                                   cfg.varPar)

        del(aryPngData)

        # Debugging feature:
        # np.save('/home/john/Desktop/aryPixConv.npy', aryPixConv)
        # *********************************************************************

        # *********************************************************************
        # *** Create pRF time courses models

        print('------Create pRF time course models')

        aryPrfTc = crt_prf_tcmdl(aryPixConv,
                                 tplVslSpcSze=cfg.tplVslSpcSze,
                                 varNumX=cfg.varNumX,
                                 varNumY=cfg.varNumY,
                                 varExtXmin=cfg.varExtXmin,
                                 varExtXmax=cfg.varExtXmax,
                                 varExtYmin=cfg.varExtYmin,
                                 varExtYmax=cfg.varExtYmax,
                                 varPrfStdMin=cfg.varPrfStdMin,
                                 varPrfStdMax=cfg.varPrfStdMax,
                                 varNumPrfSizes=cfg.varNumPrfSizes,
                                 varPar=cfg.varPar)
        # *********************************************************************

        # *********************************************************************
        # *** Save pRF time course models

        print('------Save pRF time course models to disk')

        # Save the 4D array as '*.npy' file:
        np.save(cfg.strPathMdl,
                aryPrfTc)

        # Save 4D array as '*.nii' file (for debugging purposes):
        niiPrfTc = nb.Nifti1Image(aryPrfTc, np.eye(4))
        nb.save(niiPrfTc, cfg.strPathMdl)
        # *********************************************************************

    else:

        # *********************************************************************
        # *** Load existing pRF time course models

        print('------Load pRF time course models from disk')

        # Load the file:
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model matrix has the expected
        # dimensions:
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for correct dimensions:
        lgcDim = ((vecPrfTcShp[0] == cfg.varNumX)
                  and
                  (vecPrfTcShp[1] == cfg.varNumY)
                  and
                  (vecPrfTcShp[2] == cfg.varNumPrfSizes)
                  and
                  (vecPrfTcShp[3] == cfg.varNumVol))

        # Only fit pRF models if dimensions of pRF time course models are
        # correct:
        if not(lgcDim):
            # Error message:
            strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                         'models do not agree with specified model parameters')
            raise ValueError(strErrMsg)
    # *************************************************************************

    return aryPrfTc
