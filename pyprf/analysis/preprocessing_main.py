# -*- coding: utf-8 -*-
"""Preprocessing of fMRI data and pRF model time courses."""

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
from pyprf.analysis.utilities import load_nii
from pyprf.analysis.preprocessing_par import pre_pro_par


def pre_pro_func(strPathNiiMask, lstPathNiiFunc, lgcLinTrnd=True,
                 varSdSmthTmp=2.0, varSdSmthSpt=0.0, varPar=10.0):
    """
    Load & preprocess functional data.

    Parameters
    ----------
    strPathNiiMask: str
        Path or mask used to restrict pRF model finding. Only voxels with
        a value greater than zero in the mask are considered.
    lstPathNiiFunc : list
        List of paths of functional data (nii files).
    lgcLinTrnd : bool
        Whether to perform linear trend removal on functional data.
    varSdSmthTmp : float
        Extent of temporal smoothing that is applied to functional data and
        pRF time course models, [SD of Gaussian kernel, in seconds]. If `zero`,
        no temporal smoothing is applied.
     varSdSmthSpt : float
        Extent of spatial smoothing [SD of Gaussian kernel, in mm]. If `zero`,
        no spatial smoothing is applied.
    varPar : int
        Number of processes to run in parallel (multiprocessing).

    Returns
    -------
    aryLgcMsk : np.array
        3D numpy array with logial values. Externally supplied mask (e.g grey
        matter mask). Voxels that are `False` in the mask are excluded.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of mask nii data.
    aryLgcVar : np.array
        1D numpy array containing logical values. One value per voxel after
        mask has been applied. If `True`, the variance of the voxel's time
        course is larger than zero, and the voxel is included in the output
        array (`aryFunc`). If `False`, the varuance of the voxel's time course
        is zero, and the voxel has been excluded from the output (`aryFunc`).
        This is to avoid problems in the subsequent model fitting. This array
        is necessary to put results into original dimensions after model
        fitting.
    aryFunc : np.array
        2D numpy array containing preprocessed functional data, of the form
        aryFunc[voxelCount, time].
    tplNiiShp : tuple
        Spatial dimensions of input nii data (number of voxels in x, y, z
        direction). The data are reshaped during preprocessing, this
        information is needed to fit final output into original spatial
        dimensions.

    Notes
    -----
    Functional data is loaded from disk. Temporal and spatial smoothing can be
    applied. The functional data is reshaped, into the form aryFunc[voxel,
    time]. A mask is applied (externally supplied, e.g. a grey matter mask).
    Subsequently, the functional data is de-meaned, and intensities are
    converted into z-scores.
    """
    print('------Load & preprocess nii data')

    # Load mask (to restrict model fitting):
    aryMask, hdrMsk, aryAff = load_nii(strPathNiiMask)

    # Mask is loaded as float32, but is better represented as integer:
    aryMask = np.array(aryMask).astype(np.int16)

    # Number of non-zero voxels in mask:
    # varNumVoxMsk = int(np.count_nonzero(aryMask))

    # Dimensions of nii data:
    tplNiiShp = aryMask.shape

    # Total number of voxels:
    varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Reshape mask:
    aryMask = np.reshape(aryMask, varNumVoxTlt)

    # List for arrays with functional data (possibly several runs):
    lstFunc = []

    # Number of runs:
    varNumRun = len(lstPathNiiFunc)

    # Loop through runs and load data:
    for idxRun in range(varNumRun):

        print(('---------Preprocess run ' + str(idxRun + 1)))

        # Load 4D nii data:
        aryTmpFunc, _, _ = load_nii(lstPathNiiFunc[idxRun])

        # Dimensions of nii data (including temporal dimension; spatial
        # dimensions need to be the same for mask & functional data):
        tplNiiShp = aryTmpFunc.shape

        # Preprocessing of nii data:
        aryTmpFunc = pre_pro_par(aryTmpFunc,
                                 aryMask=aryMask,
                                 lgcLinTrnd=lgcLinTrnd,
                                 varSdSmthTmp=varSdSmthTmp,
                                 varSdSmthSpt=varSdSmthSpt,
                                 varPar=varPar)

        # Reshape functional nii data, from now on of the form
        # aryTmpFunc[voxelCount, time]:
        aryTmpFunc = np.reshape(aryTmpFunc, [varNumVoxTlt, tplNiiShp[3]])

        # Apply mask:
        aryLgcMsk = np.greater(aryMask.astype(np.int16),
                               np.array([0], dtype=np.int16)[0])
        aryTmpFunc = aryTmpFunc[aryLgcMsk, :]

        # De-mean functional data:
        aryTmpFunc = np.subtract(aryTmpFunc,
                                 np.mean(aryTmpFunc,
                                         axis=1,
                                         dtype=np.float32)[:, None])

        # Convert intensities into z-scores. If there are several pRF runs,
        # these are concatenated. Z-scoring ensures that differences in mean
        # image intensity and/or variance between runs do not confound the
        # analysis. Possible enhancement: Explicitly model across-runs variance
        # with a nuisance regressor in the GLM.
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

        # Put preprocessed functional data of current run into list:
        lstFunc.append(aryTmpFunc)
        del(aryTmpFunc)

    # Put functional data from separate runs into one array. 2D array of the
    # form aryFunc[voxelCount, time]
    aryFunc = np.concatenate(lstFunc, axis=1).astype(np.float32, copy=False)
    del(lstFunc)

    # Voxels that are outside the brain and have no, or very little, signal
    # should not be included in the pRF model finding. We take the variance
    # over time and exclude voxels with a suspiciously low variance. Because
    # the data given into the cython or GPU function has float32 precision, we
    # calculate the variance on data with float32 precision.
    aryFuncVar = np.var(aryFunc, axis=1, dtype=np.float32)

    # Is the variance greater than zero?
    aryLgcVar = np.greater(aryFuncVar,
                           np.array([0.0001]).astype(np.float32)[0])

    # Array with functional data for which conditions (mask inclusion and
    # cutoff value) are fullfilled:
    aryFunc = aryFunc[aryLgcVar, :]

    return aryLgcMsk, hdrMsk, aryAff, aryLgcVar, aryFunc, tplNiiShp


def pre_pro_models(aryPrfTc, varSdSmthTmp=2.0, varPar=10):
    """
    Preprocess pRF model time courses.

    Parameters
    ----------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    varSdSmthTmp : float
        Extent of temporal smoothing that is applied to functional data and
        pRF time course models, [SD of Gaussian kernel, in seconds]. If `zero`,
        no temporal smoothing is applied.
    varPar : int
        Number of processes to run in parallel (multiprocessing).

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with preprocessed pRF time course models, same
        dimensions as input (`aryPrfTc[x-position, y-position, SD, volume]`).

    Notes
    -----
    Only temporal smoothing is applied to the pRF model time courses.
    """
    print('------Preprocess pRF time course models')
    # Preprocessing of pRF time course models:
    aryPrfTc = pre_pro_par(aryPrfTc,
                           aryMask=np.array([]),
                           lgcLinTrnd=False,
                           varSdSmthTmp=varSdSmthTmp,
                           varSdSmthSpt=0.0,
                           varPar=varPar)

    return aryPrfTc
