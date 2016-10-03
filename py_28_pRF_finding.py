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

print('---pRF analysis')

# *****************************************************************************
# *** Define parameters

# Number of x-positions to model:
varNumX = 40
# Number of y-positions to model:
varNumY = 40
# Number of pRF sizes to model:
varNumPrfSizes = 40

# Extend of visual space from centre of the screen (i.e. from the fixation
# point) [degrees of visual angle]:
varExtXmin = -5.19
varExtXmax = 5.19
varExtYmin = -5.19
varExtYmax = 5.19

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 0.20
varPrfStdMax = 7.0

# Volume TR of input data [s]:
varTr = 2.940

# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.7

# Extent of temporal smoothing for fMRI data and pRF time course models
# [standard deviation of the Gaussian kernel, in seconds]:
varSdSmthTmp = 2.940

# Extent of spatial smoothing for fMRI data [standard deviation of the Gaussian
# kernel, in mm]
varSdSmthSpt = 0.7

# Number of fMRI volumes and png files to load:
varNumVol = 400

# Intensity cutoff value for fMRI time series. Voxels with a mean intensity
# lower than the value specified here are not included in the pRF model finding
# (this speeds up the calculation, and, more importatnly, avoids division by
# zero):
varIntCtf = 50.0

# Number of processes to run in parallel:
varPar = 10

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcHighSze = (200, 200)

# Path of functional data (needs to have same number of volumes as there are
# PNGs):
strPathNiiFunc = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/nii_distcor/func_regAcrssRuns_cube/func_07.nii.gz'

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/nii_distcor/retinotopy_cython2/mask/crudebrainmask.nii.gz'

# Output basename:
strPathOut = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/nii_distcor/retinotopy_cython2/pRF_results/pRF_results'

# Create pRF time course models?
lgcCrteMdl = True

# Use cython (i.e. compiled code) for faster performance? (Requires cython to
# be installed.)
lgcCython = True

if lgcCrteMdl:
    # If we create new pRF time course models, the following parameters have to
    # be provided:

    # Size of png files (pixel*pixel):
    tplPngSize = (150, 150)

    # Basename of the 'binary stimulus files'. The files need to be in png
    # format and number in the order of their presentation during the
    # experiment.
    strPathPng = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/nii_distcor/retinotopy_cython2/pRF_stimuli/Renamed_'

    # Output path for pRF time course models file (without file extension):
    strPathMdl = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/nii_distcor/retinotopy_cython2/pRF_results/pRF_model_tc'

else:
    # If we use existing pRF time course models, the path to the respective
    # file has to be provided (including file extension, i.e. '*.npy'):
    strPathMdl = '.npy'
# *****************************************************************************


# *****************************************************************************
# *** Import modules

import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import scipy as sp
import nibabel as nb
import time
import multiprocessing as mp
from scipy.stats import gamma
from scipy.interpolate import griddata
from py_32_pRF_filtering import funcPrfPrePrc
if lgcCython:
	from py_42_cython_lstsqr import funcCyLsq
# *****************************************************************************


# *****************************************************************************
# ***  Check time
varTme01 = time.time()
# *****************************************************************************


# *****************************************************************************
# *** Determine CPU access

# Get the process PID:
# varPid = os.getpid()
# Prepare command:
# strTmp = ('taskset -cp 0-' + str(varPar) + ' ' + str(varPid))
# Issue command:
# os.system(strTmp)
# affinity.set_process_affinity_mask(varPid, 2**varPar)
# *****************************************************************************

# *****************************************************************************
# *** Preparations

# Convert preprocessing parameters (for temporal and spatial smoothing) from
# SI units (i.e. [s] and [mm]) into units of data array:
varSdSmthTmp = np.divide(varSdSmthTmp, varTr)
varSdSmthSpt = np.divide(varSdSmthSpt, varVoxRes)
# *****************************************************************************


# *****************************************************************************
# ***  Define functions

def funcGauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """Create 2D Gaussian kernel."""
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # aryX and aryY are in reversed order, this seems to be necessary:
    aryY, aryX = sp.mgrid[0:varSizeX,
                          0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (
            np.power((aryX - varPosX), 2.0) +
            np.power((aryY - varPosY), 2.0)
        ) /
        (2.0 * np.power(varSd, 2.0))
        )
    aryGauss = np.exp(-aryGauss)

    return aryGauss


def funcHrf(varNumVol, varTr):
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


def funcConvPar(vecDm, vecHrf, varNumVol, idxX, idxY, queOut):
    """Convolution of pixel-wise 'design matrix' with HRF model."""
    # In order to avoid an artefact at the end of the time series, we have to
    # concatenate an empty array to both the design matrix and the HRF model
    # before convolution.
    vecZeros = np.zeros([100, 1]).flatten()
    vecDm = np.concatenate((vecDm, vecZeros))
    vecHrf = np.concatenate((vecHrf, vecZeros))

    # Convolve design matrix with HRF model:
    vecConv = np.convolve(vecDm, vecHrf, mode='full')[0:varNumVol]

    # Create list containing the convolved design matrix, and the X and Y pixel
    # values to which this design matrix corresponds to:
    lstOut = [idxX, idxY, vecConv]

    # Put output to queue:
    queOut.put(lstOut)


def funcPrfTc(aryMdlParamsChnk, tplVslSpcHighSze, varNumVol, aryPngDataHigh,
              queOut):
    """Create pRF time course models."""
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
        # in other words, the pRF time course model.
        aryPrfTcTmp = np.divide(aryPrfTcTmp,
                                np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idxMdl, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    aryOut = np.hstack((np.array(aryMdlParamsChnk[:, 0], ndmin=2).T,
                        aryOut))

    # Put output to queue:
    queOut.put(aryOut)


def funcFindPrf(idxPrc, varNumX, varNumY, varNumPrfSizes, vecMdlXpos,
                vecMdlYpos, vecMdlSd, aryFuncChnk, aryPrfTc, queOut):
    """Find the best pRF model for voxel time course."""
    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    # varNumVol = aryFuncChnk.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk), 100000000.0).astype(np.float32)

    # Vector that will hold the temporary residuals from the model fitting:
    # vecTmpRes = np.zeros(varNumVoxChnk).astype(np.float32)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Prepare data for cython (i.e. accelerated) least squares finding:
    if lgcCython:
        # Instead of fitting a constant term, we subtract the mean from the
        # data and from the model ("FSL style") First, we subtract the mean
        # over time from the data:
        aryFuncChnkTmean = np.array(np.mean(aryFuncChnk, axis=0), ndmin=2)
        aryFuncChnk = np.subtract(aryFuncChnk, aryFuncChnkTmean[0, None])
        # Secondly, we subtract the mean over time form the pRF model time
        # courses. The array has four dimensions, the 4th is time (one to three
        # are x-position, y-position, and pRF size (SD)).
        aryPrfTcTmean = np.mean(aryPrfTc, axis=3)
        aryPrfTc = np.subtract(aryPrfTc, aryPrfTcTmean[:, :, :, None])
    # Otherwise, create constant term for numpy least squares finding:
    else:
        # Constant term for the model:
        vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumMdls = (varNumX * varNumY * varNumPrfSizes)

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrf = np.ceil(vecStatPrf)
        vecStatPrf = vecStatPrf.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through pRF models:
    for idxX in range(0, varNumX):

        for idxY in range(0, varNumY):

            for idxSd in range(0, varNumPrfSizes):

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Status indicator:
                    if varCntSts02 == vecStatPrf[varCntSts01]:

                        # Prepare status message:
                        strStsMsg = ('---------Progress: ' +
                                     str(vecStatPrc[varCntSts01]) +
                                     ' % --- ' +
                                     str(vecStatPrf[varCntSts01]) +
                                     ' pRF models out of ' +
                                     str(varNumMdls))

                        print(strStsMsg)

                        # Only increment counter if the last value has not been
                        # reached yet:
                        if varCntSts01 < varStsStpSze:
                            varCntSts01 = varCntSts01 + int(1)

                # Calculation of the ratio of the explained variance (R square)
                # for the current model for all voxel time courses.

                # Cython version:
                if lgcCython:

                    # A cython function is used to calculate the residuals
                    # of the current model:
                    vecTmpRes = funcCyLsq(
                        aryPrfTc[idxX, idxY, idxSd, :].flatten(), aryFuncChnk)

                # Numpy version:
                else:

                    # Current pRF time course model:
                    vecMdlTc = aryPrfTc[idxX, idxY, idxSd, :].flatten()

                    # We create a design matrix including the current pRF time
                    # course model, and a constant term:
                    aryDsgn = np.vstack([vecMdlTc,
                                         vecConst]).T

                    # Change type to float32:
                    aryDsgn = aryDsgn.astype(np.float32)

                    # Calculate the least-squares solution for all voxels:
                    vecTmpRes = np.linalg.lstsq(aryDsgn, aryFuncChnk)[1]

                # Check whether current residuals are lower than previously
                # calculated ones:
                vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

                # Replace best x and y position values, and SD values.
                vecBstXpos[vecLgcTmpRes] = vecMdlXpos[idxX]
                vecBstYpos[vecLgcTmpRes] = vecMdlYpos[idxY]
                vecBstSd[vecLgcTmpRes] = vecMdlSd[idxSd]

                # Replace best residual values:
                vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

                # Status indicator (only used in the first of the parallel
                # processes):
                if idxPrc == 0:

                    # Increment status indicator counter:
                    varCntSts02 = varCntSts02 + 1

    # After finding the best fitting model for each voxel, we still have to
    # calculate the coefficient of determination (R-squared) for each voxel. We
    # start by calculating the total sum of squares (i.e. the deviation of the
    # data from the mean). The mean of each time course:
    vecFuncMean = np.mean(aryFuncChnk, axis=0)
    # Deviation from the mean for each datapoint:
    vecFuncDev = np.subtract(aryFuncChnk, vecFuncMean[None, :])
    # Sum of squares:
    vecSsTot = np.sum(np.power(vecFuncDev,
                               2.0),
                      axis=0)
    # Coefficient of determination:
    vecBstR2 = np.subtract(1.0,
                           np.divide(vecBstRes,
                                     vecSsTot))

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              vecBstR2]

    queOut.put(lstOut)
# *****************************************************************************


# *****************************************************************************
# ***  Create new pRF time course models, or load existing models

if lgcCrteMdl:

    # Create new pRF time course models

    # *************************************************************************
    # *** Load PNGs

    print('------Load PNGs')

    # Create list of png files to load:
    lstPngPaths = [None] * varNumVol
    for idx01 in range(0, varNumVol):
        lstPngPaths[idx01] = (strPathPng + str(idx01) + '.png')

    # Load png files. The png data will be saved in a numpy array of the
    # following order: aryPngData[x-pixel, y-pixel, PngNumber]. The
    # sp.misc.imread function actually contains three values per pixel (RGB),
    # but since the stimuli are black-and-white, any one of these is sufficient
    # and we discard the others.
    aryPngData = np.zeros((tplPngSize[0], tplPngSize[1], varNumVol))
    for idx01 in range(0, varNumVol):
        aryPngData[:, :, idx01] = sp.misc.imread(lstPngPaths[idx01])[:, :, 0]

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 0).astype(int)
    # *************************************************************************

    # *************************************************************************
    # *** Create pixel-wise HRF model time courses

    print('------Create pixel-wise HRF model time courses')

    # Create HRF time course:
    vecHrf = funcHrf(varNumVol, varTr)

    # Empty list for results of convolution (pixel-wise model time courses):
    lstPixConv = [None] * (tplPngSize[0] * tplPngSize[1])

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Counter for parallel processes:
    varCntPar = 0
    # Counter for output of parallel processes:
    varCntOut = 0

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Convolve pixel-wise 'design matrix' with HRF. For loops through X and Y
    # position:
    for idxX in range(0, tplPngSize[0]):

        for idxY in range(0, tplPngSize[1]):

            # Create process:
            lstPrcs[varCntPar] = mp.Process(target=funcConvPar,
                                            args=(aryPngData[idxX, idxY, :],
                                                  vecHrf,
                                                  varNumVol,
                                                  idxX,
                                                  idxY,
                                                  queOut)
                                            )

            # Daemon (kills processes when exiting):
            lstPrcs[varCntPar].Daemon = True

            # Check whether it's time to start & join process (exception for
            # first volume):
            if \
                np.mod(int(varCntPar + 1), int(varPar)) == \
                    int(0) and int(varCntPar) > 1:

                # Start processes:
                for idxPrc in range(0, varPar):
                    lstPrcs[idxPrc].start()

                # Collect results from queue:
                for idxPrc in range(0, varPar):
                    lstPixConv[varCntOut] = queOut.get(True)
                    # Increment output counter:
                    varCntOut = varCntOut + 1

                # Join processes:
                for idxPrc in range(0, varPar):
                    lstPrcs[idxPrc].join()

                # Reset process counter:
                varCntPar = 0

                # Create empty list:
                lstPrcs = [None] * varPar
                # queOut = mp.Queue()

            else:
                # Increment process counter:
                varCntPar = varCntPar + 1

    # Final round of processes:
    if varCntPar != 0:

        # Start processes:
        for idxPrc in range(0, varCntPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(0, varCntPar):
            lstPixConv[varCntOut] = queOut.get(True)
            # Increment output counter:
            varCntOut = varCntOut + 1

        # Join processes:
        for idxPrc in range(0, varCntPar):
            lstPrcs[idxPrc].join()

    # Array for convolved pixel-wise HRF model time courses, of the form
    # aryPixConv[x-position, y-position, volume]:
    aryPixConv = np.zeros([tplPngSize[0], tplPngSize[1], varNumVol])

    # Put convolved pixel-wise HRF model time courses into array (they
    # originally needed to be saved in a list due to parallelisation). Each
    # entry in the list holds three items: the x-position of the respective
    # pixel, the y-position of  the respective pixel, and a vector with the
    # model time course.
    for idxLst in range(0, len(lstPixConv)):
        # Load the list corresponding to the current voxel from the parent
        # list:
        lstTmp = lstPixConv[idxLst]
        # Access x-position and y-position:
        varTmpPosX = lstTmp[0]
        varTmpPosY = lstTmp[1]
        # Put current pixel's model time course into array:
        aryPixConv[varTmpPosX, varTmpPosY, :] = lstTmp[2]

    # Delete the large pixel time course list:
    del(lstPixConv)

    # # Debugging feature:
    # aryPixConv = np.around(aryPixConv, 3)
    # for idxVol in range(0, varNumVol):
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
    aryPngDataHigh = np.zeros((tplVslSpcHighSze[0],
                               tplVslSpcHighSze[1],
                               varNumVol))

    # Loop through volumes:
    for idxVol in range(0, varNumVol):

        # The following array describes the coordinates of the pixels in the
        # flattened array (i.e. "vecOrigPixVal"). In other words, these are the
        # row and column coordinates of the original pizel values.
        aryOrixPixCoo = np.zeros([int(tplPngSize[0] * tplPngSize[1]), 2])

        # Range for the coordinates:
        vecRange = np.arange(0, tplPngSize[0])

        # X coordinates:
        vecCooX = np.repeat(vecRange, tplPngSize[0])

        # Y coordinates:
        vecCooY = np.tile(vecRange, tplPngSize[1])

        # Put the pixel coordinates into the respective array:
        aryOrixPixCoo[:, 0] = vecCooX
        aryOrixPixCoo[:, 1] = vecCooY

        # The following vector will contain the actual original pixel values:
        # vecOrigPixVal = np.zeros([1, int(tplPngSize[0] * tplPngSize[1])])
        vecOrigPixVal = aryPixConv[:, :, idxVol]
        vecOrigPixVal = vecOrigPixVal.flatten()

        # The sampling interval for the creation of the super-sampled pixel
        # data (complex numbers are used as a convention for inclusive
        # intervals in "np.mgrid()").:
        # varStpSzeX = float(tplPngSize[0]) / float(tplVslSpcHighSze[0])
        # varStpSzeY = float(tplPngSize[1]) / float(tplVslSpcHighSze[1])
        varStpSzeX = np.complex(tplVslSpcHighSze[0])
        varStpSzeY = np.complex(tplVslSpcHighSze[1])

        # The following grid has the coordinates of the points at which we
        # would like to re-sample the pixel data:
        aryPixGridX, aryPixGridY = np.mgrid[0:tplPngSize[0]:varStpSzeX,
                                            0:tplPngSize[1]:varStpSzeY]

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
    # The pRF time course models are created using the super-sampled model of
    # the pixel time courses.

    print('------Create pRF time course models')

    # Upsampling factor:
    if (tplVslSpcHighSze[0] / varNumX) == (tplVslSpcHighSze[1] / varNumY):
        varFctUp = tplVslSpcHighSze[0] / varNumX
    else:
        print('------ERROR. Dimensions of upsampled visual space do not ' +
              'agree with specified number of pRFs to model.')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0,
                       (tplVslSpcHighSze[0] - 1),
                       varNumX,
                       endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0,
                       (tplVslSpcHighSze[1] - 1),
                       varNumY,
                       endpoint=True)

    # Vector with the standard deviations of the pRF models. We need to convert
    # the standard deviation values from degree of visual angle to the
    # dimensions of the visual space. We calculate the scaling factor from
    # degrees of visual angle to pixels in the *upsampled* visual space
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixUpX = tplVslSpcHighSze[0] / (varExtXmax - varExtXmin)
    varDgr2PixUpY = tplVslSpcHighSze[1] / (varExtYmax - varExtYmin)

    # The factor relating pixels in the upsampled visual space to degrees of
    # visual angle should be roughly the same (allowing for some rounding error
    # if the visual stimulus was not square):
    if 0.5 < np.absolute((varDgr2PixUpX - varDgr2PixUpY)):
        print('------ERROR. The ratio of X and Y dimensions in stimulus ' +
              'space (in degrees of visual angle) and the ratio of X and Y ' +
              'dimensions in the upsampled visual space do not agree')

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin,
                           varPrfStdMax,
                           varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector with the pRF sizes to be modelled with the scaling
    # factor (for the x-dimensions - as we have just found out, the scaling
    # factors for the x- and y-direction are identical, except for rounding
    # error). Now the vector with the pRF sizes to be modelled is can directly
    # be used for the creation of Gaussian pRF models in upsampled visual
    # space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixUpX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = [None] * varPar

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumMdls,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumMdls))

    # Put model parameters into chunks:
    for idxChnk in range(0, varPar):
        # Index of first combination of model parameters to be included in
        # current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last combination of model parameters to be included in
        # current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstMdlParams[idxChnk] = aryMdlParams[varTmpChnkSrt:varTmpChnkEnd, :]

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstPrfTc = [None] * varPar

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcPrfTc,
                                     args=(lstMdlParams[idxPrc],
                                           tplVslSpcHighSze,
                                           varNumVol,
                                           aryPngDataHigh,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    # test:
    #    ary1 = np.array([np.arange(0,5), np.arange(0,5)])
    #    ary3 = np.array([np.arange(5,10), np.arange(5,10)])
    #    ary2 = np.array([np.arange(10,15), np.arange(10,15)])
    #    lst1 = [ary1, ary2, ary3]
    #    ary4 = np.vstack(lst1)
    #    ary5 = ary4[np.argsort(ary4[:, 0])]

    # Put output arrays from parallel process into one big array (where each
    # row corresponds to one model time course, the first column corresponds to
    # the index number of the model time course, and the remaining columns
    # correspond to time points):
    aryPrfTc = np.vstack(lstPrfTc)

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Sort output along the first column (which contains the indicies), so that
    # the output is in the same order as the list of combination of model
    # parameters which we created before the parallelisation:
    aryPrfTc = aryPrfTc[np.argsort(aryPrfTc[:, 0])]

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryPrfTc4D = np.zeros([varNumX, varNumY, varNumPrfSizes, varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryPrfTc4D[idxX, idxY, idxSd, :] = aryPrfTc[varCntMdlPrms, 1:]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # Change array name for consistency, and delete unnecessary copy:
    aryPrfTc = np.copy(aryPrfTc4D)
    del(aryPrfTc4D)
    # *************************************************************************

    # *************************************************************************
    # *** Save pRF time course models

    # Save the 4D array as '*.npy' file:
    np.save(strPathMdl,
            aryPrfTc)

    # Save 4D array as '*.nii' file (for debugging purposes):
    niiPrfTc = nb.Nifti1Image(aryPrfTc, np.eye(4))
    nb.save(niiPrfTc, strPathMdl)

    # Set test for correct dimensions of '*.npy' file to true:
    lgcDim = True
    # *************************************************************************

else:

    # *************************************************************************
    # *** Load existing pRF time course models

    print('------Load pRF time course models')

    # Load the file:
    aryPrfTc = np.load(strPathMdl)

    # Check whether pRF time course model matrix has the expected dimensions:
    vecPrfTcShp = aryPrfTc.shape

    # Logical test for correct dimensions:
    lgcDim = ((vecPrfTcShp[0] == varNumX)
              and
              (vecPrfTcShp[1] == varNumY)
              and
              (vecPrfTcShp[2] == varNumPrfSizes)
              and
              (vecPrfTcShp[3] == varNumVol))
# *****************************************************************************


# *****************************************************************************
# *** Find pRF models for voxel time courses

# Only fit pRF models if dimensions of pRF time course models are correct:
if lgcDim:

    print('------Find pRF models for voxel time courses')

    print('---------Loading nii data')

    # Load 4D nii data:
    niiFunc = nb.load(strPathNiiFunc)
    # Load the data into memory:
    aryFunc = niiFunc.get_data()
    aryFunc = np.array(aryFunc)

    # Load mask (to restrict model fining):
    niiMask = nb.load(strPathNiiMask)
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
                                      varSdSmthTmp,
                                      varSdSmthSpt,
                                      varIntCtf,
                                      varPar)

    # Number of non-zero voxels in mask (i.e. number of voxels for which pRF
    # finding will be performed):
    varNumMskVox = int(np.count_nonzero(aryMask))

    # Dimensions of nii data:
    vecNiiShp = aryFunc.shape

    print('---------Preparing parallel pRF model finding')

    # Vector with the moddeled x-positions of the pRFs:
    vecMdlXpos = np.linspace(varExtXmin,
                             varExtXmax,
                             varNumX,
                             endpoint=True)

    # Vector with the moddeled y-positions of the pRFs:
    vecMdlYpos = np.linspace(varExtYmin,
                             varExtYmax,
                             varNumY,
                             endpoint=True)

    # Vector with the moddeled standard deviations of the pRFs:
    vecMdlSd = np.linspace(varPrfStdMin,
                           varPrfStdMax,
                           varNumPrfSizes,
                           endpoint=True)

    # Empty list for results (parameters of best fitting pRF model):
    lstPrfRes = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Counter for parallel processes:
    varCntPar = 0

    # Counter for output of parallel processes:
    varCntOut = 0

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
                         np.greater(aryFuncMean, varIntCtf))

    # Array with functional data for which conditions (mask inclusion and
    # cutoff value) are fullfilled:
    aryFunc = aryFunc[aryLgc, :]

    # Number of voxels for which pRF finding will be performed:
    varNumVoxInc = aryFunc.shape[0]

    print('---------Number of voxels on which pRF finding will be ' +
          'performed: ' + str(varNumVoxInc))

    # List into which the chunks of functional data for the parallel processes
    # will be put:
    lstFunc = [None] * varPar

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumVoxInc,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumVoxInc))

    # Put functional data into chunks:
    for idxChnk in range(0, varPar):
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
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=funcFindPrf,
                                     args=(idxPrc,
                                           varNumX,
                                           varNumY,
                                           varNumPrfSizes,
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
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Prepare pRF finding results for export')

    # Create list for vectors with fitting results, in order to put the results
    # into the correct order:
    lstResXpos = [None] * varPar
    lstResYpos = [None] * varPar
    lstResSd = [None] * varPar
    lstResR2 = [None] * varPar

    # Put output into correct order:
    for idxRes in range(0, varPar):

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
    for idxRes in range(0, varPar):
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
        strTmp = (strPathOut + lstNiiNames[idxOut] + '.nii')
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
