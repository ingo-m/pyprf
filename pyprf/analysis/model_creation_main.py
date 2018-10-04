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
import h5py
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
                              cfg.lstPathPng,
                              cfg.tplVslSpcSze,
                              varStrtIdx=cfg.varStrtIdx,
                              varZfill=cfg.varZfill)
        # *********************************************************************

        # *********************************************************************
        # *** Convolve pixel-wise design matrix with HRF model

        print('------Convolve pixel-wise design matrix with HRF model')

        # Debugging feature:
        # np.save('/home/john/Desktop/aryPngData.npy', aryPngData)

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

        # Size of model parameter space:
        varSzeMdlSpc = (cfg.varNumX
                        * cfg.varNumY
                        * cfg.varNumPrfSizes
                        * cfg.varNumCon)

        # If the number of volumes is large (multi-run experiment) or the
        # size of the model parameter space is large, the pRF time course
        # models will not fit into RAM. In this case, they are stored in an
        # hdf5 file (location specified by 'strPathMdl', as specified in the
        # config file).
        lgcHdf5 = ((1000 < cfg.varNumVol) or (100000 < varSzeMdlSpc))
        if lgcHdf5:
            # If model space is large, pass filepath to model creation
            # function.
            strPathMdlTmp = cfg.strPathMdl
        else:
            # If the file
            strPathMdlTmp = None

        aryPrfTc = crt_prf_tcmdl(aryPixConv,
                                 strPathMdl=strPathMdlTmp,
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

        if lgcHdf5:

            # In case of hdf5 mode, the pRF time courses should have already
            # been written to disk (from the parallelised child processes). So
            # no need to save them here. But a nii version is saved for visual
            # inspection.

            # Save model time courses as '*.nii' file - hdf5 mode. We load and
            # save one stimulus condition at a time, in order to prevent
            # out of memory.

            # Path of hdf5 file:
            strPthHdf5 = (cfg.strPathMdl + '.hdf5')

            # Read file:
            fleHdf5 = h5py.File(strPthHdf5, 'r')

            # Access dataset in current hdf5 file:
            aryPrfTc = fleHdf5['pRF_time_courses']

        else:

            print('------Save pRF time course models to disk')

            # Array with pRF time course models, shape:
            # aryPrfTc[x-position, y-position, SD, condition, volume].

            # Save the 5D array as '*.npy' file:
            np.save(cfg.strPathMdl,
                    aryPrfTc)

        # Save model time courses as '*.nii' file (for debugging purposes).
        # Nii file can be inspected visually, e.g. using fsleyes. We save
        # one 4D nii file per stimulus condition.
        varNumCon = aryPrfTc.shape[3]
        for idxCon in range(varNumCon):
            niiPrfTc = nb.Nifti1Image(aryPrfTc[:, :, :, idxCon, :],
                                      np.eye(4))
            nb.save(niiPrfTc,
                    (cfg.strPathMdl + '_condition_' + str(idxCon)))
        # *********************************************************************

    else:

        # *********************************************************************
        # *** Load existing pRF time course models

        print('------Load pRF time course models from disk')

        # Load the file. Array with pRF time course models, shape:
        # aryPrfTc[x-position, y-position, SD, condition, volume].
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model array has the expected
        # dimensions.
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for dimensions of parameter space:
        lgcDim = ((vecPrfTcShp[0] == cfg.varNumX)
                  and
                  (vecPrfTcShp[1] == cfg.varNumY)
                  and
                  (vecPrfTcShp[2] == cfg.varNumPrfSizes))

        # Only fit pRF models if dimensions of pRF time course models are
        # as expected.
        strErrMsg = ('Dimensions of specified pRF time course models do not '
                     + 'agree with specified model parameters')
        assert lgcDim, strErrMsg
        # *********************************************************************

    return aryPrfTc
