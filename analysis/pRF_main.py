# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields."""


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
import scipy as sp
import nibabel as nb
import time
import multiprocessing as mp
from scipy.interpolate import griddata

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
# SI units (i.e. [s] and [mm]) into units of data array:
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
        if idx01 < 9:
            lstPngPaths[idx01] = (cfg.strPathPng + '00' + str(idx01 + 1) +
                                  '.png')
        elif idx01 < 99:
            lstPngPaths[idx01] = (cfg.strPathPng + '0' + str(idx01 + 1) +
                                  '.png')
        elif idx01 < 999:
            lstPngPaths[idx01] = (cfg.strPathPng + str(idx01 + 1) + '.png')

    # Load png files. The png data will be saved in a numpy array of the
    # following order: aryPngData[x-pixel, y-pixel, PngNumber]. The
    # sp.misc.imread function actually contains three values per pixel (RGB),
    # but since the stimuli are black-and-white, any one of these is sufficient
    # and we discard the others.
    aryPngData = np.zeros((cfg.tplPngSize[0],
                           cfg.tplPngSize[1],
                           cfg.varNumVol))
    for idx01 in range(0, cfg.varNumVol):
        # aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :, 0]
        aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :]

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 200).astype(int)
    # *************************************************************************

    # *************************************************************************
    # *** Create pixel-wise HRF model time courses

    print('------Create pixel-wise HRF model time courses')

    aryPixConv = funcCrtPixMdl(aryPngData,
                               cfg.varNumVol,
                               cfg.varTr,
                               cfg.tplPngSize,
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
    # *** Resample pixel-time courses in high-res visual space
    # The Gaussian sampling of the pixel-time courses takes place in the
    # super-sampled visual space. Here we take the convolved pixel-time courses
    # into this space, for each time point (volume).

    print('------Resample pixel-time courses in high-res visual space')

    # Array for super-sampled pixel-time courses:
    aryPngDataHigh = np.zeros((cfg.tplVslSpcHighSze[0],
                               cfg.tplVslSpcHighSze[1],
                               cfg.varNumVol))

    # Loop through volumes:
    for idxVol in range(0, cfg.varNumVol):

        # The following array describes the coordinates of the pixels in the
        # flattened array (i.e. "vecOrigPixVal"). In other words, these are the
        # row and column coordinates of the original pizel values.
        aryOrixPixCoo = np.zeros([int(cfg.tplPngSize[0] * cfg.tplPngSize[1]),
                                  2])

        # Range for the coordinates:
        vecRange = np.arange(0, cfg.tplPngSize[0])

        # X coordinates:
        vecCooX = np.repeat(vecRange, cfg.tplPngSize[0])

        # Y coordinates:
        vecCooY = np.tile(vecRange, cfg.tplPngSize[1])

        # Put the pixel coordinates into the respective array:
        aryOrixPixCoo[:, 0] = vecCooX
        aryOrixPixCoo[:, 1] = vecCooY

        # The following vector will contain the actual original pixel values:
        # vecOrigPixVal = np.zeros([1,
        #                           int(cfg.tplPngSize[0]
        #                               * cfg.tplPngSize[1])])
        vecOrigPixVal = aryPixConv[:, :, idxVol]
        vecOrigPixVal = vecOrigPixVal.flatten()

        # The sampling interval for the creation of the super-sampled pixel
        # data (complex numbers are used as a convention for inclusive
        # intervals in "np.mgrid()").:
        # varStpSzeX = (float(cfg.tplPngSize[0])
        #               / float(cfg.tplVslSpcHighSze[0]))
        # varStpSzeY = (float(cfg.tplPngSize[1])
        #               / float(cfg.tplVslSpcHighSze[1]))
        varStpSzeX = np.complex(cfg.tplVslSpcHighSze[0])
        varStpSzeY = np.complex(cfg.tplVslSpcHighSze[1])

        # The following grid has the coordinates of the points at which we
        # would like to re-sample the pixel data:
        aryPixGridX, aryPixGridY = np.mgrid[0:cfg.tplPngSize[0]:varStpSzeX,
                                            0:cfg.tplPngSize[1]:varStpSzeY]

        # The actual resampling:
        aryResampled = griddata(aryOrixPixCoo,
                                vecOrigPixVal,
                                (aryPixGridX, aryPixGridY),
                                method='nearest')

        # Put super-sampled pixel time courses into array:
        aryPngDataHigh[:, :, idxVol] = aryResampled
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
                               aryPngDataHigh)
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

    print('------Find pRF models for voxel time courses')

    print('---------Loading nii data')

    # Load 4D nii data:
    niiFunc = nb.load(cfg.strPathNiiFunc)
    # Load the data into memory:
    aryFunc = niiFunc.get_data()
    aryFunc = np.array(aryFunc)

    # Load mask (to restrict model fining):
    niiMask = nb.load(cfg.strPathNiiMask)
    # Get nii header of mask:
    hdrMsk = niiMask.header
    # Get nii 'affine':
    affMsk = niiMask.affine
    # Load the data into memory:
    aryMask = niiMask.get_data()
    aryMask = np.array(aryMask)

    # Preprocessing of fMRI data and pRF time course models:
    aryFunc, aryPrfTc = funcPrfPrePrc(aryFunc,
                                      aryMask,
                                      aryPrfTc,
                                      cfg.varSdSmthTmp,
                                      cfg.varSdSmthSpt,
                                      cfg.varIntCtf,
                                      cfg.varPar)

    # Number of non-zero voxels in mask (i.e. number of voxels for which pRF
    # finding will be performed):
    varNumMskVox = int(np.count_nonzero(aryMask))

    # Dimensions of nii data:
    vecNiiShp = aryFunc.shape

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

    # Take mean over time of functional nii data:
    aryFuncMean = np.mean(aryFunc, axis=1)

    # Logical test for voxel inclusion: is the voxel value greater than zero in
    # the mask, and is the mean of the functional time series above the cutoff
    # value?
    aryLgc = np.multiply(np.greater(aryMask, 0),
                         np.greater(aryFuncMean, cfg.varIntCtf))

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
