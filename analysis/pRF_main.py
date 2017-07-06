# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields.

Use `import pRF_config as cfg` for static pRF analysis.

Use `import pRF_config_motion as cfg` for pRF analysis with motion stimuli.
"""


# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# *****************************************************************************
# *** Import modules

# import os
# os.chdir(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
#import scipy as sp
import nibabel as nb
import time
import multiprocessing as mp
#from scipy.interpolate import griddata
from PIL import Image

import pRF_config as cfg
from pRF_crtPixMdl import funcCrtPixMdl
from pRF_funcFindPrf import funcFindPrf
from pRF_filtering import funcPrfPrePrc
from pRF_crtPrfTcMdl import funcCrtPrfTcMdl
# *****************************************************************************


# *****************************************************************************
# *** Check time
print('---pRF analysis')
varTme01 = time.time()
# *****************************************************************************


# *****************************************************************************
# *** Determine CPU access

# Get the process PID:
# varPid = os.getpid()
# Prepare command:
# strTmp = ('taskset -cp 0-' + str(cfg.varPar) + ' ' + str(varPid))
# Issue command:
# os.system(strTmp)
# affinity.set_process_affinity_mask(varPid, 2**cfg.varPar)
# *****************************************************************************


# *****************************************************************************
# *** Preparations

# Convert preprocessing parameters (for temporal and spatial smoothing) from
# SI units (i.e. [s] and [mm]) into units of data array (volumes and voxels):
cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)
cfg.varSdSmthSpt = np.divide(cfg.varSdSmthSpt, cfg.varVoxRes)

# Compile cython code:
# ...
# *****************************************************************************


# *****************************************************************************
# *** Create new pRF time course models, or load existing models

if cfg.lgcCrteMdl:  #noqa

    # Create new pRF time course models

    # *************************************************************************
    # *** Load PNGs

    print('------Load PNGs')

    # Create list of png files to load:
    lstPngPaths = [None] * cfg.varNumVol
    for idx01 in range(0, cfg.varNumVol):
        lstPngPaths[idx01] = (cfg.strPathPng
                              + str(idx01 + 1).zfill(3)
                              + '.png')

    # Open first image to check PNG size:
    # objIm = Image.open(lstPngPaths[0])
    # tplPngSize = objIm.size

    # Load png files. The png data will be saved in a numpy array of the
    # following order: aryPngData[x-pixel, y-pixel, PngNumber]. The
    # sp.misc.imread function actually contains three values per pixel (RGB),
    # but since the stimuli are black-and-white, any one of these is sufficient
    # and we discard the others.

    aryPngData = np.zeros((cfg.tplVslSpcHighSze[0],
                           cfg.tplVslSpcHighSze[1],
                           cfg.varNumVol))

    for idx01 in range(0, cfg.varNumVol):

        # Old version of reading images with scipy
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :, 0]
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :]

        # Load & resize image:
        objIm = Image.open(lstPngPaths[idx01])
        objIm = objIm.resize((cfg.tplVslSpcHighSze[0],
                              cfg.tplVslSpcHighSze[1]),
                             resample=Image.NEAREST)
        aryPngData[:, :, idx01] = np.array(objIm.getdata()).reshape( \
            objIm.size[0], objIm.size[1])

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 200).astype(np.int8)
    # *************************************************************************


    # *************************************************************************
    # *** Create pixel-wise HRF model time courses

    print('------Create pixel-wise HRF model time courses')

    aryPixConv = funcCrtPixMdl(aryPngData,
                               cfg.varNumVol,
                               cfg.varTr,
                               cfg.tplVslSpcHighSze,
                               cfg.varPar)

    # # Debugging feature:
    # aryPixConv = np.around(aryPixConv, 3)
    # for idxVol in range(0, cfg.varNumVol):
    #     strTmp = ('/home/john/Desktop/png_test/png_vol_' +
    #               str(idxVol) +
    #               '.png')
    #     # imsave(strTmp, (aryPixConv[:, :, idxVol] * 100))
    #     toimage((aryPixConv[:, :, idxVol] * 100),
    #                cmin=-5,
    #                cmax=105).save(strTmp)
    # *************************************************************************


    # *************************************************************************
    # *** Create pRF time courses models

    print('------Create pRF time course models')

    # The pRF time course models are created using the super-sampled model of
    # the pixel time courses.
    aryPrfTc = funcCrtPrfTcMdl(cfg.tplVslSpcHighSze,
                               cfg.varNumX,
                               cfg.varNumY,
                               cfg.varExtXmin,
                               cfg.varExtXmax,
                               cfg.varExtYmin,
                               cfg.varExtYmax,
                               cfg.varPrfStdMin,
                               cfg.varPrfStdMax,
                               cfg.varNumPrfSizes,
                               cfg.varPar,
                               cfg.varNumVol,
                               aryPngData)
    # *************************************************************************

    # *************************************************************************
    # *** Save pRF time course models

    # Save the 4D array as '*.npy' file:
    np.save(cfg.strPathMdl,
            aryPrfTc)

    # Save 4D array as '*.nii' file (for debugging purposes):
    niiPrfTc = nb.Nifti1Image(aryPrfTc, np.eye(4))
    nb.save(niiPrfTc, cfg.strPathMdl)

    # Set test for correct dimensions of '*.npy' file to true:
    lgcDim = True
    # *************************************************************************

else:

    # *************************************************************************
    # *** Load existing pRF time course models

    print('------Load pRF time course models')

    # Load the file:
    aryPrfTc = np.load(cfg.strPathMdl)

    # Check whether pRF time course model matrix has the expected dimensions:
    vecPrfTcShp = aryPrfTc.shape

    # Logical test for correct dimensions:
    lgcDim = ((vecPrfTcShp[0] == cfg.varNumX)
              and
              (vecPrfTcShp[1] == cfg.varNumY)
              and
              (vecPrfTcShp[2] == cfg.varNumPrfSizes)
              and
              (vecPrfTcShp[3] == cfg.varNumVol))
# *****************************************************************************


# *****************************************************************************
# *** Find pRF models for voxel time courses

# Only fit pRF models if dimensions of pRF time course models are correct:
if lgcDim:

    print('---------Load & preprocess nii data')

    # Load mask (to restrict model fitting):
    niiMask = nb.load(cfg.strPathNiiMask)
    # Get nii header of mask:
    hdrMsk = niiMask.header
    # Get nii 'affine':
    affMsk = niiMask.affine
    # Load the data into memory:
    aryMask = niiMask.get_data()
    aryMask = np.array(aryMask)

    # List for arrays with functional datas for runs:
    lstFunc = []

    # Number of runs:
    varNumRun = len(cfg.lstPathNiiFunc)

    # Loop through runs and load data:
    for idxRun in range(varNumRun):

        print(('------Preprocess run ' + str(idxRun + 1)))

        # Load 4D nii data:
        niiTmpFunc = nb.load(cfg.lstPathNiiFunc[idxRun])
        # Load the data into memory:
        aryTmpFunc = niiTmpFunc.get_data()
        aryTmpFunc = np.array(aryTmpFunc)

        # Preprocessing of nii data:
        aryTmpFunc = funcPrfPrePrc(aryTmpFunc,
                                   aryMask=aryMask,
                                   lgcLinTrnd=True,
                                   varSdSmthTmp=cfg.varSdSmthTmp,
                                   varSdSmthSpt=cfg.varSdSmthSpt,
                                   varIntCtf=cfg.varIntCtf,
                                   varPar=cfg.varPar)

        # Demeaning (runs are concatenated, therefore we demean) - POSSIBLE
        # ENHANCEMENT: don't demean runs, but model across-runs variance
        # explicitly in GLM.
        aryTmpMne = np.mean(aryTmpFunc, axis=3)
        aryTmpFunc = np.subtract(aryTmpFunc,
                                 aryTmpMne[:, :, :, None])

        # Put preprocessed functional data of current run into list:
        lstFunc.append(np.copy(aryTmpFunc))

    # Put functional data from separate runs into one array:
    aryFunc = np.concatenate(lstFunc, axis=3)
    del(lstFunc)
    del(aryTmpFunc)

    print('---------Preprocess pRF time course models')

    # Preprocessing of pRF time course models:
    aryPrfTc = funcPrfPrePrc(aryPrfTc,
                             aryMask=np.array([]),
                             lgcLinTrnd=False,
                             varSdSmthTmp=cfg.varSdSmthTmp,
                             varSdSmthSpt=0.0,
                             varIntCtf=0.0,
                             varPar=cfg.varPar)


    # Number of non-zero voxels in mask (i.e. number of voxels for which pRF
    # finding will be performed):
    varNumMskVox = int(np.count_nonzero(aryMask))

    # Dimensions of nii data:
    vecNiiShp = aryFunc.shape

    print('------Find pRF models for voxel time courses')

    print('---------Preparing parallel pRF model finding')

    # Vector with the moddeled x-positions of the pRFs:
    vecMdlXpos = np.linspace(cfg.varExtXmin,
                             cfg.varExtXmax,
                             cfg.varNumX,
                             endpoint=True)

    # Vector with the moddeled y-positions of the pRFs:
    vecMdlYpos = np.linspace(cfg.varExtYmin,
                             cfg.varExtYmax,
                             cfg.varNumY,
                             endpoint=True)

    # Vector with the moddeled standard deviations of the pRFs:
    vecMdlSd = np.linspace(cfg.varPrfStdMin,
                           cfg.varPrfStdMax,
                           cfg.varNumPrfSizes,
                           endpoint=True)

    # Empty list for results (parameters of best fitting pRF model):
    lstPrfRes = [None] * cfg.varPar

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Total number of voxels:
    varNumVoxTlt = (vecNiiShp[0] * vecNiiShp[1] * vecNiiShp[2])

    # Reshape functional nii data:
    aryFunc = np.reshape(aryFunc, [varNumVoxTlt, vecNiiShp[3]])

    # Reshape mask:
    aryMask = np.reshape(aryMask, varNumVoxTlt)


    # Voxels that are outside the brain and have no, or very little, signal
    # should not be included in the pRF model finding. We take the variance
    # over time and exclude voxels with a suspiciously low variance. Because
    # the data given into the cython function has float32 precision, we
    # calculate the variance on data with float32 precision.
    aryFuncVar = np.var(aryFunc.astype(np.float32), axis=1)

    # Logical test for voxel inclusion: is the voxel value greater than zero in
    # the mask, and is the variance greater than zero?
    aryLgc = np.multiply(np.greater(aryMask, 0),
                         np.greater(aryFuncVar,
                                    np.array(([0.0001])).astype(np.float32))
                         )

    # Array with functional data for which conditions (mask inclusion and
    # cutoff value) are fullfilled:
    aryFunc = aryFunc[aryLgc, :]

    # Number of voxels for which pRF finding will be performed:
    varNumVoxInc = aryFunc.shape[0]

    print('---------Number of voxels on which pRF finding will be ' +
          'performed: ' + str(varNumVoxInc))

    # List into which the chunks of functional data for the parallel processes
    # will be put:
    lstFunc = [None] * cfg.varPar

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumVoxInc,
                              num=cfg.varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumVoxInc))

    # Put functional data into chunks:
    for idxChnk in range(0, cfg.varPar):
        # Index of first voxel to be included in current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last voxel to be included in current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstFunc[idxChnk] = aryFunc[varTmpChnkSrt:varTmpChnkEnd, :]

    # We don't need the original array with the functional data anymore:
    del(aryFunc)

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcFindPrf,
                                     args=(idxPrc,
                                           cfg.varNumX,
                                           cfg.varNumY,
                                           cfg.varNumPrfSizes,
                                           vecMdlXpos,
                                           vecMdlYpos,
                                           vecMdlSd,
                                           lstFunc[idxPrc],
                                           aryPrfTc,
                                           cfg.lgcCython,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, cfg.varPar):
        lstPrfRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc].join()

    print('---------Prepare pRF finding results for export')

    # Create list for vectors with fitting results, in order to put the results
    # into the correct order:
    lstResXpos = [None] * cfg.varPar
    lstResYpos = [None] * cfg.varPar
    lstResSd = [None] * cfg.varPar
    lstResR2 = [None] * cfg.varPar

    # Put output into correct order:
    for idxRes in range(0, cfg.varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstPrfRes[idxRes][0]

        # Put fitting results into list, in correct order:
        lstResXpos[varTmpIdx] = lstPrfRes[idxRes][1]
        lstResYpos[varTmpIdx] = lstPrfRes[idxRes][2]
        lstResSd[varTmpIdx] = lstPrfRes[idxRes][3]
        lstResR2[varTmpIdx] = lstPrfRes[idxRes][4]

    # Concatenate output vectors (into the same order as the voxels that were
    # included in the fitting):
    aryBstXpos = np.zeros(0)
    aryBstYpos = np.zeros(0)
    aryBstSd = np.zeros(0)
    aryBstR2 = np.zeros(0)
    for idxRes in range(0, cfg.varPar):
        aryBstXpos = np.append(aryBstXpos, lstResXpos[idxRes])
        aryBstYpos = np.append(aryBstYpos, lstResYpos[idxRes])
        aryBstSd = np.append(aryBstSd, lstResSd[idxRes])
        aryBstR2 = np.append(aryBstR2, lstResR2[idxRes])

    # Delete unneeded large objects:
    del(lstPrfRes)
    del(lstResXpos)
    del(lstResYpos)
    del(lstResSd)
    del(lstResR2)

    # Array for pRF finding results, of the form
    # aryPrfRes[total-number-of-voxels, 0:3], where the 2nd dimension
    # contains the parameters of the best-fitting pRF model for the voxel, in
    # the order (0) pRF-x-pos, (1) pRF-y-pos, (2) pRF-SD, (3) pRF-R2.
    aryPrfRes = np.zeros((varNumVoxTlt, 6))

    # Put results form pRF finding into array (they originally needed to be
    # saved in a list due to parallelisation).
    aryPrfRes[aryLgc, 0] = aryBstXpos
    aryPrfRes[aryLgc, 1] = aryBstYpos
    aryPrfRes[aryLgc, 2] = aryBstSd
    aryPrfRes[aryLgc, 3] = aryBstR2

    # Reshape pRF finding results:
    aryPrfRes = np.reshape(aryPrfRes,
                           [vecNiiShp[0],
                            vecNiiShp[1],
                            vecNiiShp[2],
                            6])

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

    # Save nii results:
    for idxOut in range(0, 6):
        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryPrfRes[:, :, :, idxOut],
                                affMsk,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = (cfg.strPathOut + lstNiiNames[idxOut] + '.nii')
        nb.save(niiOut, strTmp)
    # *************************************************************************

else:
    # Error message:
    strErrMsg = ('---Error: Dimensions of specified pRF time course models ' +
                 'do not agree with specified model parameters')
    print(strErrMsg)

# *****************************************************************************
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
# *****************************************************************************
