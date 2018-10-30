# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields.

Use `import pRF_config as cfg` for static pRF analysis.

Use `import pRF_config_motion as cfg` for pRF analysis with motion stimuli.
"""


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

import time
import numpy as np
import nibabel as nb
import h5py

from pyprf.analysis.load_config import load_config
from pyprf.analysis.utilities import cls_set_config
from pyprf.analysis.model_creation_main import model_creation
from pyprf.analysis.preprocessing_main import pre_pro_models
from pyprf.analysis.preprocessing_main import pre_pro_func

from pyprf.analysis.preprocessing_hdf5 import pre_pro_models_hdf5
from pyprf.analysis.preprocessing_hdf5 import pre_pro_func_hdf5

from pyprf.analysis.find_prf import find_prf


def pyprf(strCsvCnfg, lgcTest=False):  #noqa
    """
    Main function for pRF mapping.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.
    """
    # *************************************************************************
    # *** Check time
    print('---pRF analysis')
    varTme01 = time.time()
    # *************************************************************************

    # *************************************************************************
    # *** Preparations

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Convert preprocessing parameters (for temporal and spatial smoothing)
    # from SI units (i.e. [s] and [mm]) into units of data array (volumes and
    # voxels):
    cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)
    cfg.varSdSmthSpt = np.divide(cfg.varSdSmthSpt, cfg.varVoxRes)

    # For the GPU version, we need to set down the parallelisation to 1 now,
    # because no separate CPU threads are to be created. We may still use CPU
    # parallelisation for preprocessing, which is why the parallelisation
    # factor is only reduced now, not earlier.
    if cfg.strVersion == 'gpu':
        cfg.varPar = 1
    # *************************************************************************

    # *************************************************************************
    # *** Create or load pRF time course models

    # In case of a multi-run experiment, the data may not fit into memory.
    # (Both pRF model time courses and the fMRI data may be large in this
    # case.) Therefore, we switch to hdf5 mode, where model time courses and
    # fMRI data are hold in hdf5 files (on disk). The location of the hdf5 file
    # for model time courses is specified by 'strPathMdl' (in the config file).
    # The hdf5 file with fMRI data are stored at the same location as the input
    # nii files.

    # Array with pRF time course models, shape:
    # aryPrfTc[x-position, y-position, SD, condition, volume].
    # If in hdf5 mode, `aryPrfTc` is `None`.
    aryPrfTc = model_creation(dicCnfg, lgcHdf5=cfg.lgcHdf5)
    # *************************************************************************

    # *************************************************************************
    # *** Preprocessing

    if cfg.lgcHdf5:

        print('---Hdf5 mode.')

        # Preprocessing of functional data:
        vecLgcMsk, hdrMsk, aryAff, vecLgcVar, tplNiiShp, strPthHdf5Func = \
            pre_pro_func_hdf5(cfg.strPathNiiMask,
                              cfg.lstPathNiiFunc,
                              lgcLinTrnd=cfg.lgcLinTrnd,
                              varSdSmthTmp=cfg.varSdSmthTmp,
                              varSdSmthSpt=cfg.varSdSmthSpt)

        # Preprocessing of pRF model time courses:
        strPrfTc, aryLgcMdlVar = \
            pre_pro_models_hdf5(cfg.strPathMdl,
                                varSdSmthTmp=cfg.varSdSmthTmp,
                                strVersion=cfg.strVersion,
                                varPar=cfg.varPar)

        # Dummy pRF time courses (for compatibility with regular mode):
        aryPrfTc = None

        # ---Makeshift solution for small data after masking---

        # TODO: IMPLEMENT FULL HDF5 MODE FOR READING OF FUNCTIONAL DATA.

        # Read hdf5 file (masked timecourses of current run):
        fleHdfFunc = h5py.File(strPthHdf5Func, 'r')

        # Access dataset in current hdf5 file:
        dtsFunc = fleHdfFunc['func']
        aryFunc = dtsFunc[:, :]
        aryFunc = np.copy(aryFunc)
        fleHdfFunc.close()

    else:

        # Preprocessing of pRF model time courses:
        aryPrfTc = pre_pro_models(aryPrfTc,
                                  varSdSmthTmp=cfg.varSdSmthTmp,
                                  varPar=cfg.varPar)

        # Preprocessing of functional data:
        vecLgcMsk, hdrMsk, aryAff, vecLgcVar, aryFunc, tplNiiShp = \
            pre_pro_func(cfg.strPathNiiMask,
                         cfg.lstPathNiiFunc,
                         lgcLinTrnd=cfg.lgcLinTrnd,
                         varSdSmthTmp=cfg.varSdSmthTmp,
                         varSdSmthSpt=cfg.varSdSmthSpt,
                         varPar=cfg.varPar)

        # Dummy variables (for compatibility with hdf5 mode):
        strPrfTc = None
        aryLgcMdlVar = None
    # *************************************************************************

    # *************************************************************************
    # *** Find pRF models for voxel time courses.

    lstPrfRes = find_prf(dicCnfg, aryFunc, aryPrfTc=aryPrfTc,
                         aryLgcMdlVar=aryLgcMdlVar, strPrfTc=strPrfTc)
    # *************************************************************************

    # *************************************************************************
    # *** Merge results from parallel processes

    print('---------Prepare pRF finding results for export')

    # Create list for vectors with fitting results, in order to put the results
    # into the correct order:
    lstResXpos = [None] * cfg.varPar
    lstResYpos = [None] * cfg.varPar
    lstResSd = [None] * cfg.varPar
    lstResR2 = [None] * cfg.varPar
    lstResPe = [None] * cfg.varPar

    # Put output into correct order:
    for idxRes in range(cfg.varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstPrfRes[idxRes][0]

        # Put fitting results into list, in correct order:
        lstResXpos[varTmpIdx] = lstPrfRes[idxRes][1]
        lstResYpos[varTmpIdx] = lstPrfRes[idxRes][2]
        lstResSd[varTmpIdx] = lstPrfRes[idxRes][3]
        lstResR2[varTmpIdx] = lstPrfRes[idxRes][4]
        lstResPe[varTmpIdx] = lstPrfRes[idxRes][5]

    # Concatenate output vectors (into the same order as the voxels that were
    # included in the fitting):
    aryBstXpos = np.concatenate(lstResXpos, axis=0).astype(np.float32)
    aryBstYpos = np.concatenate(lstResYpos, axis=0).astype(np.float32)
    aryBstSd = np.concatenate(lstResSd, axis=0).astype(np.float32)
    aryBstR2 = np.concatenate(lstResR2, axis=0).astype(np.float32)
    # aryBstXpos = np.zeros(0, dtype=np.float32)
    # aryBstYpos = np.zeros(0, dtype=np.float32)
    # aryBstSd = np.zeros(0, dtype=np.float32)
    # aryBstR2 = np.zeros(0, dtype=np.float32)
    # for idxRes in range(0, cfg.varPar):
    #     aryBstXpos = np.append(aryBstXpos, lstResXpos[idxRes])
    #     aryBstYpos = np.append(aryBstYpos, lstResYpos[idxRes])
    #     aryBstSd = np.append(aryBstSd, lstResSd[idxRes])
    #     aryBstR2 = np.append(aryBstR2, lstResR2[idxRes])

    # Concatenate PEs, shape: aryBstPe[varNumVox, varNumCon].
    aryBstPe = np.concatenate(lstResPe, axis=0).astype(np.float32)
    varNumCon = aryBstPe.shape[1]

    # Delete unneeded large objects:
    del(lstPrfRes)
    del(lstResXpos)
    del(lstResYpos)
    del(lstResSd)
    del(lstResR2)
    del(lstResPe)
    # *************************************************************************

    # *************************************************************************
    # *** Reshape spatial parameters

    # Put results form pRF finding into array (they originally needed to be
    # saved in a list due to parallelisation). Voxels were selected for pRF
    # model finding in two stages: First, a mask was applied. Second, voxels
    # with low variance were removed. Voxels are put back into the original
    # format accordingly.

    # Number of voxels that were included in the mask:
    varNumVoxMsk = np.sum(vecLgcMsk)

    # Array for pRF finding results, of the form aryPrfRes[voxel-count, 0:3],
    # where the 2nd dimension contains the parameters of the best-fitting pRF
    # model for the voxel, in the order (0) pRF-x-pos, (1) pRF-y-pos, (2)
    # pRF-SD, (3) pRF-R2. At this step, only the voxels included in the mask
    # are represented.
    aryPrfRes01 = np.zeros((varNumVoxMsk, 6), dtype=np.float32)

    # Place voxels based on low-variance exlusion:
    aryPrfRes01[vecLgcVar, 0] = aryBstXpos
    aryPrfRes01[vecLgcVar, 1] = aryBstYpos
    aryPrfRes01[vecLgcVar, 2] = aryBstSd
    aryPrfRes01[vecLgcVar, 3] = aryBstR2

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, 6), dtype=np.float32)
    aryPrfRes02[vecLgcMsk, 0] = aryPrfRes01[:, 0]
    aryPrfRes02[vecLgcMsk, 1] = aryPrfRes01[:, 1]
    aryPrfRes02[vecLgcMsk, 2] = aryPrfRes01[:, 2]
    aryPrfRes02[vecLgcMsk, 3] = aryPrfRes01[:, 3]

    # Reshape pRF finding results into original image dimensions:
    aryPrfRes = np.reshape(aryPrfRes02,
                           [tplNiiShp[0],
                            tplNiiShp[1],
                            tplNiiShp[2],
                            6])

    del(aryPrfRes01)
    del(aryPrfRes02)
    # *************************************************************************

    # *************************************************************************
    # *** Reshape parameter estimates (betas)

    # Bring PEs into original data shape. First, account for binary (brain)
    # mask:
    aryPrfRes01 = np.zeros((varNumVoxMsk, varNumCon), dtype=np.float32)

    # Place voxels based on low-variance exlusion:
    aryPrfRes01[vecLgcVar, :] = aryBstPe

    # Place voxels based on mask-exclusion:
    aryPrfRes02 = np.zeros((varNumVoxTlt, varNumCon), dtype=np.float32)
    aryPrfRes02[vecLgcMsk, :] = aryPrfRes01

    # Reshape pRF finding results into original image dimensions:
    aryBstPe = np.reshape(aryPrfRes02,
                          [tplNiiShp[0],
                           tplNiiShp[1],
                           tplNiiShp[2],
                           varNumCon])

    # New shape: aryBstPe[x, y, z, varNumCon]

    del(aryPrfRes01)
    del(aryPrfRes02)
    # *************************************************************************

    # *************************************************************************
    # *** Export results

    # The nii header of the mask will be used for creation of result nii files.
    # Set dtype to float32 to avoid precision loss (in case mask is int).
    hdrMsk.set_data_dtype(np.float32)

    # Calculate polar angle map:
    aryPrfRes[:, :, :, 4] = np.arctan2(aryPrfRes[:, :, :, 1],
                                       aryPrfRes[:, :, :, 0])

    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryPrfRes[:, :, :, 5] = np.sqrt(np.add(np.power(aryPrfRes[:, :, :, 0],
                                                    2.0),
                                           np.power(aryPrfRes[:, :, :, 1],
                                                    2.0)))

    # List with name suffices of output images:
    lstNiiNames = ['_x_pos',
                   '_y_pos',
                   '_SD',
                   '_R2',
                   '_polar_angle',
                   '_eccentricity']

    print('---------Exporting results')

    # Save spatial pRF parameters to nii:
    for idxOut in range(6):
        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryPrfRes[:, :, :, idxOut],
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = (cfg.strPathOut + lstNiiNames[idxOut] + '.nii.gz')
        nb.save(niiOut, strTmp)

    # Save PEs to nii (not implemented for gpu mode):
    if cfg.strVersion != 'gpu':
        for idxCon in range(varNumCon):
            # Create nii object for results:
            niiOut = nb.Nifti1Image(aryBstPe[:, :, :, idxCon],
                                    aryAff,
                                    header=hdrMsk
                                    )
            # Save nii:
            strTmp = (cfg.strPathOut
                      + '_PE_'
                      + str(idxCon + 1).zfill(2)
                      + '.nii.gz')
            nb.save(niiOut, strTmp)
    # *************************************************************************

    # *************************************************************************
    # *** Report time

    varTme02 = time.time()
    varTme03 = varTme02 - varTme01
    print('---Elapsed time: ' + str(varTme03) + ' s')
    print('---Done.')
    # *************************************************************************
