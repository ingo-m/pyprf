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


# *****************************************************************************
# *** Import modules

import sys
import numpy as np
import nibabel as nb
import time
import multiprocessing as mp
from PIL import Image

# import pRF_config as cfg
import pRF_config_motion as cfg

from pRF_utilities import fncLoadNii, fncLoadLargeNii
from pRF_crtPixMdl import funcCrtPixMdl
from pRF_funcFindPrf import funcFindPrf
from pRF_funcFindPrfGpuQ import funcFindPrfGpu
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
        aryPngData[:, :, idx01] = np.array(objIm.getdata()).reshape(
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

    # Only fit pRF models if dimensions of pRF time course models are correct:
    if not(lgcDim):
        # Error message:
        strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                     'models do not agree with specified model parameters')
        raise ValueError(strErrMsg)
# *****************************************************************************


# *****************************************************************************
# *** Preprocessing

print('------Load & preprocess nii data')

# Load mask (to restrict model fitting):
aryMask, hdrMsk, affMsk = fncLoadNii(cfg.strPathNiiMask)

# Mask is loaded as float32, but is better represented as integer:
aryMask = np.array(aryMask).astype(np.int16)

# Number of non-zero voxels in mask:
# varNumVoxMsk = int(np.count_nonzero(aryMask))

# Dimensions of nii data:
vecNiiShp = aryMask.shape

# Total number of voxels:
varNumVoxTlt = (vecNiiShp[0] * vecNiiShp[1] * vecNiiShp[2])

# Reshape mask:
aryMask = np.reshape(aryMask, varNumVoxTlt)

# List for arrays with functional data (possibly several runs):
lstFunc = []

# Number of runs:
varNumRun = len(cfg.lstPathNiiFunc)

# Loop through runs and load data:
for idxRun in range(varNumRun):

    print(('------Preprocess run ' + str(idxRun + 1)))

    # Load 4D nii data:
    aryTmpFunc, _, _ = fncLoadLargeNii(cfg.lstPathNiiFunc[idxRun])

    # Dimensions of nii data (including temporal dimension; spatial dimensions
    # need to be the same for mask & functional data):
    vecNiiShp = aryTmpFunc.shape

    # Preprocessing of nii data:
    aryTmpFunc = funcPrfPrePrc(aryTmpFunc,
                               aryMask=aryMask,
                               lgcLinTrnd=True,
                               varSdSmthTmp=cfg.varSdSmthTmp,
                               varSdSmthSpt=cfg.varSdSmthSpt,
                               varIntCtf=cfg.varIntCtf,
                               varPar=cfg.varPar)

    # Reshape functional nii data, from now on of the form
    # aryTmpFunc[voxelCount, time]:
    aryTmpFunc = np.reshape(aryTmpFunc, [varNumVoxTlt, vecNiiShp[3]])

    # Convert intensities into z-scores. If there are several pRF runs, these
    # are concatenated. Z-scoring ensures that differences in mean image
    # intensity and/or variance between runs do not confound the analysis.
    # Possible enhancement: Explicitly model across-runs variance with a
    # nuisance regressor in the GLM.
    aryTmpStd = np.std(aryTmpFunc, axis=1)

    # In order to avoid devision by zero, only divide those voxels with a
    # standard deviation greater than zero:
    aryTmpLgc = np.greater(aryTmpStd.astype(np.float32),
                           np.array([0.0], dtype=np.float32)[0])
    # Z-scoring:
    aryTmpFunc[aryTmpLgc, :] = np.divide(aryTmpFunc[aryTmpLgc, :],
                                         aryTmpStd[aryTmpLgc, None])
    # Set voxels with a variance of zero to intensity zero:
    aryTmpLgc = np.not_equal(aryTmpLgc, True)
    aryTmpFunc[aryTmpLgc, :] = np.array([0.0], dtype=np.float32)[0]

    # Apply mask:
    aryLgcMsk = np.greater(aryMask.astype(np.int16),
                           np.array([0], dtype=np.int16)[0])
    aryTmpFunc = aryTmpFunc[aryLgcMsk, :]

    # Put preprocessed functional data of current run into list:
    lstFunc.append(np.copy(aryTmpFunc))
    del(aryTmpFunc)

# Put functional data from separate runs into one array. 2D array of the form
# aryFunc[voxelCount, time]
aryFunc = np.concatenate(lstFunc, axis=1).astype(np.float32, copy=False)
del(lstFunc)

# Voxels that are outside the brain and have no, or very little, signal should
# not be included in the pRF model finding. We take the variance over time and
# exclude voxels with a suspiciously low variance. Because the data given into
# the cython or GPU function has float32 precision, we calculate the variance
# on data with float32 precision.
aryFuncVar = np.var(aryFunc, axis=1, dtype=np.float32)

# Is the variance greater than zero?
aryLgcVar = np.greater(aryFuncVar,
                       np.array([0.0001]).astype(np.float32)[0])

# Array with functional data for which conditions (mask inclusion and cutoff
# value) are fullfilled:
aryFunc = aryFunc[aryLgcVar, :]

# Number of voxels for which pRF finding will be performed:
varNumVoxInc = aryFunc.shape[0]

print('---------Number of voxels on which pRF finding will be performed: '
      + str(varNumVoxInc))

print('---------Preprocess pRF time course models')

# Preprocessing of pRF time course models:
aryPrfTc = funcPrfPrePrc(aryPrfTc,
                         aryMask=np.array([]),
                         lgcLinTrnd=False,
                         varSdSmthTmp=cfg.varSdSmthTmp,
                         varSdSmthSpt=0.0,
                         varIntCtf=0.0,
                         varPar=cfg.varPar)
# *****************************************************************************


# *****************************************************************************
# *** Find pRF models for voxel time courses

print('------Find pRF models for voxel time courses')

print('---------Preparing parallel pRF model finding')

# For the GPU version, we need to set down the parallelisation to 1 now,
# because no separate CPU threads are to be created. We may still use CPU
# parallelisation for preprocessing, which is why the parallelisation factor is
# only reduced now, not earlier.
if cfg.strVersion == 'gpu':
    cfg.varPar = 1

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

# List into which the chunks of functional data for the parallel processes will
# be put:
lstFunc = [None] * cfg.varPar

# Vector with the indicies at which the functional data will be separated in
# order to be chunked up for the parallel processes:
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

# CPU version (using numpy or cython for pRF finding):
if ((cfg.strVersion == 'numpy') or (cfg.strVersion == 'cython')):

    print('---------pRF finding on CPU')

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
                                           cfg.strVersion,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

# CPU version (using numpy or cython for pRF finding):
elif cfg.strVersion == 'gpu':

    print('---------pRF finding on GPU')

    # Create processes:
    for idxPrc in range(0, cfg.varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcFindPrfGpu,
                                     args=(idxPrc,
                                           cfg.varNumX,
                                           cfg.varNumY,
                                           cfg.varNumPrfSizes,
                                           vecMdlXpos,
                                           vecMdlYpos,
                                           vecMdlSd,
                                           lstFunc[idxPrc],
                                           aryPrfTc,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, cfg.varPar):
    lstPrcs[idxPrc].start()

# Delete reference to list with function data (the data continues to exists in
# child process memory space):
del(lstFunc)

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

# Put results form pRF finding into array (they originally needed to be saved
# in a list due to parallelisation). Voxels were selected for pRF model finding
# in two stages: First, a mask was applied. Second, voxels with low variance
# were removed. Voxels are put back into the original format accordingly.

# Number of voxels that were included in the mask:
varNumVoxMsk = np.sum(aryLgcMsk)

# Array for pRF finding results, of the form aryPrfRes[voxel-count, 0:3], where
# the 2nd dimension contains the parameters of the best-fitting pRF model for
# the voxel, in the order (0) pRF-x-pos, (1) pRF-y-pos, (2) pRF-SD, (3) pRF-R2.
# At this step, only the voxels included in the mask are represented.
aryPrfRes01 = np.zeros((varNumVoxMsk, 6), dtype=np.float32)

# Place voxels based on low-variance exlusion:
aryPrfRes01[aryLgcVar, 0] = aryBstXpos
aryPrfRes01[aryLgcVar, 1] = aryBstYpos
aryPrfRes01[aryLgcVar, 2] = aryBstSd
aryPrfRes01[aryLgcVar, 3] = aryBstR2

# Place voxels based on mask-exclusion:
aryPrfRes02 = np.zeros((varNumVoxTlt, 6), dtype=np.float32)
aryPrfRes02[aryLgcMsk, 0] = aryPrfRes01[:, 0]
aryPrfRes02[aryLgcMsk, 1] = aryPrfRes01[:, 1]
aryPrfRes02[aryLgcMsk, 2] = aryPrfRes01[:, 2]
aryPrfRes02[aryLgcMsk, 3] = aryPrfRes01[:, 3]

# Reshape pRF finding results into original image dimensions:
aryPrfRes = np.reshape(aryPrfRes02,
                       [vecNiiShp[0],
                        vecNiiShp[1],
                        vecNiiShp[2],
                        6])

del(aryPrfRes01)
del(aryPrfRes02)

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
# *****************************************************************************


# *****************************************************************************
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('-Elapsed time: ' + str(varTme03) + ' s')
print('-Done.')
# *****************************************************************************
