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

import os
import numpy as np
import h5py
import threading
import queue
from scipy.ndimage.filters import gaussian_filter
from pyprf.analysis.utilities import load_nii
from pyprf.analysis.preprocessing_par import funcLnTrRm
from pyprf.analysis.preprocessing_par import funcSmthTmp
from pyprf.analysis.nii_to_hdf5 import feed_hdf5
from pyprf.analysis.nii_to_hdf5 import feed_hdf5_spt
from pyprf.analysis.nii_to_hdf5 import feed_hdf5_tme
from pyprf.analysis.preprocessing_par import pre_pro_par


def pre_pro_func_hdf5(strPathNiiMask, lstPathNiiFunc, lgcLinTrnd=True,
                      varSdSmthTmp=2.0, varSdSmthSpt=0.0):
    """
    Load & preprocess functional data - hdf5 mode.

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
    vecLgcMsk : np.array
        1D numpy array with logial values. Externally supplied mask (e.g grey
        matter mask). Voxels that are `False` in the mask are excluded.
    hdrMsk : nibabel-header-object
        Nii header of mask.
    aryAff : np.array
        Array containing 'affine', i.e. information about spatial positioning
        of mask nii data.
    vecLgcVar : np.array
        1D numpy array containing logical values. One value per voxel after
        mask has been applied. If `True`, the variance of the voxel's time
        course is larger than zero, and the voxel is included in the output
        array (`aryFunc`). If `False`, the varuance of the voxel's time course
        is zero, and the voxel has been excluded from the output (`aryFunc`).
        This is to avoid problems in the subsequent model fitting. This array
        is necessary to put results into original dimensions after model
        fitting.
    tplHdf5Shp : tuple
        Spatial dimensions of input nii data (number of voxels in x, y, z
        direction). The data are reshaped during preprocessing, this
        information is needed to fit final output into original spatial
        dimensions.
    strPthHdf5Func : str
        Path of hdf5 files with preprocessed functional data.

    Notes
    -----
    Functional data is manipulated on disk (hdf5 mode). Temporal and spatial
    smoothing can be applied. The functional data is reshaped, into the form
    aryFunc[time, voxel]. A mask is applied (externally supplied, e.g. a grey
    matter mask). Subsequently, the functional data is de-meaned, and
    intensities are converted into z-scores.

    """
    print('------Load & preprocess nii data (hdf5 mode).')

    # Load mask (to restrict model fitting):
    aryMask, hdrMsk, aryAff = load_nii(strPathNiiMask)

    # Dimensions of nii data:
    tplNiiShp = aryMask.shape

    # Total number of voxels:
    varNumVox = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])

    # Mask is loaded as float32, but is better represented as integer:
    aryMask = np.array(aryMask).astype(np.int16)

    # Reshape mask:
    aryMask = np.reshape(aryMask, varNumVox)

    # Make mask boolean:
    vecLgcMsk = np.greater(aryMask.astype(np.int16),
                           np.array([0], dtype=np.int16)[0])

    # Number of voxels after masking:
    varNumVoxMsk = np.sum(vecLgcMsk)

    # Number of runs:
    varNumRun = len(lstPathNiiFunc)

    # Counter for total number of volumes (in case number of volumes differs
    # between runs).
    varNumVolTtl = 0

    # Remember file names of masked hdf5 files.
    lstFleMsk = [None] * varNumRun

    # Loop through runs and load data:
    for idxRun in range(varNumRun):

        print(('---------Preprocess run ' + str(idxRun + 1)))

        # Path of 4D nii file:
        strPthNii = lstPathNiiFunc[idxRun]

        # File path & file name:
        strFlePth, strFleNme = os.path.split(strPthNii)

        # Remove file extension from file name:
        strFleNme = strFleNme.split('.')[0]

        # Path of hdf5 file with functional data (corresponding to 4D nii file
        # for current run).
        strPthHdf5 = os.path.join(strFlePth, (strFleNme + '.hdf5'))

        assert os.path.isfile(strPthHdf5), 'HDF5 file not found.'

        # Read & write file:
        fleHdf5 = h5py.File(strPthHdf5, 'r+')

        # Access dataset in current hdf5 file:
        dtsFunc = fleHdf5['func']

        # Dimensions of hdf5 data (should be of shape func[time, voxel]):
        tplHdf5Shp = dtsFunc.shape

        # Number of time points in hdf5 file:
        varNumVol = tplHdf5Shp[0]

        # Increment total-volume counter:
        varNumVolTtl += varNumVol

        # Preprocessing of nii data.

        # ---------------------------------------------------------------------
        # Linear trend removal

        if lgcLinTrnd:

            print('---------Linear trend removal')

            # Looping voxel by voxel is too slow. Instead, read & write a
            # chunks of voxels at a time. Indices of chunks:
            varStpSze = 100
            vecSplt = np.arange(0, (varNumVox + 1), varStpSze)
            vecSplt = np.concatenate((vecSplt, np.array([varNumVox])))

            # Number of chunks:
            varNumCnk = vecSplt.shape[0]

            # Buffer size:
            varBuff = 10

            # Create FIFO queue:
            objQ = queue.Queue(maxsize=varBuff)

            # Define & run extra thread with graph that places data on queue:
            objThrd = threading.Thread(target=feed_hdf5_spt,
                                       args=(dtsFunc, objQ, vecSplt))
            objThrd.setDaemon(True)
            objThrd.start()

            # Loop through chunks of volumes:
            for idxChnk in range((varNumCnk - 1)):

                # Start index of current chunk:
                varIdx01 = vecSplt[idxChnk]

                # Stop index of current chunk:
                varIdx02 = vecSplt[idxChnk + 1]

                # Get chunk of functional data from hdf5 file:
                aryFunc = dtsFunc[:, varIdx01:varIdx02]

                # Perform linear trend removal:
                aryFunc = funcLnTrRm(0, aryFunc, 0.0, None)

                # Put result on queue (from where it will be saved to disk in a
                # separate thread).
                objQ.put(aryFunc)

            # Close thread:
            objThrd.join()

        # ---------------------------------------------------------------------
        # Spatial smoothing

        if 0.0 < varSdSmthSpt:

            print('---------Spatial smoothing')

            # Looping volume by volume is too slow. Instead, read & write a
            # chunk of volumes at a time. Indices of chunks:
            varStpSze = 10
            vecSplt = np.arange(0, (varNumVol + 1), varStpSze)
            vecSplt = np.concatenate((vecSplt, np.array([varNumVol])))

            # Number of chunks:
            varNumCnk = vecSplt.shape[0]

            # Buffer size:
            varBuff = 10

            # Create FIFO queue:
            objQ = queue.Queue(maxsize=varBuff)

            # Define & run extra thread with graph that places data on queue:
            objThrd = threading.Thread(target=feed_hdf5_tme,
                                       args=(dtsFunc, objQ, vecSplt))
            objThrd.setDaemon(True)
            objThrd.start()

            # Loop through chunks of volumes:
            for idxChnk in range((varNumCnk - 1)):

                # Start index of current chunk:
                varIdx01 = vecSplt[idxChnk]

                # Stop index of current chunk:
                varIdx02 = vecSplt[idxChnk + 1]

                # Number of volumes in current chunk:
                # varNumVolTmp = varIdx02 - varIdx01

                # Get chunk of functional data from hdf5 file:
                aryFunc = dtsFunc[varIdx01:varIdx02, :]

                # Loop through volumes (within current chunk):
                varChnkNumVol = aryFunc.shape[0]
                for idxVol in range(varChnkNumVol):

                    # Reshape into original shape (for spatial smoothing):
                    # aryTmp = np.reshape(aryFunc[idxVol, :], [tplNiiShp[0], tplNiiShp[1], tplNiiShp[2]])
                    # aryTmp = np.reshape(aryFunc[idxVol, :], [58, 33, 25], order='C') # : {‘C’, ‘F’, ‘A’}, optional)
                    aryTmp = np.reshape(
                                        np.reshape(aryFunc[idxVol, :], (varNumVox, 1)).T,
                                        [tplNiiShp[0], tplNiiShp[1], tplNiiShp[2]])

                    # Perform smoothing:
                    aryTmp = gaussian_filter(
                        aryTmp.astype(np.float32),
                        varSdSmthSpt,
                        order=0,
                        mode='nearest',
                        truncate=4.0).astype(np.float32)

                    # Back to shape: func[time, voxel].
                    #aryFunc[idxVol, :] = aryTmp.reshape(varNumVox)
                    aryFunc[idxVol, :] = np.reshape(aryTmp, [varNumVox])

                # Put current volume on queue.
                objQ.put(aryFunc)

            # Close thread:
            objThrd.join()

        # ---------------------------------------------------------------------
        # Apply mask

        # Path of hdf5 file for masked functional data:
        strPthHdf5Msk = os.path.join(strFlePth, (strFleNme + '_masked.hdf5'))

        # Remember file names of masked hdf5 files:
        lstFleMsk[idxRun] = strPthHdf5Msk

        # Create hdf5 file:
        fleHdf5Msk = h5py.File(strPthHdf5Msk, 'w')

        # Create dataset within hdf5 file:
        dtsFuncMsk = fleHdf5Msk.create_dataset('func',
                                               (varNumVol, varNumVoxMsk),
                                               dtype=np.float32)

        # Buffer size:
        varBuff = 10

        # Create FIFO queue:
        objQ = queue.Queue(maxsize=varBuff)

        # Define & run extra thread with graph that places data on queue:
        objThrd = threading.Thread(target=feed_hdf5,
                                   args=(dtsFuncMsk, objQ, varNumVoxMsk))
        objThrd.setDaemon(True)
        objThrd.start()

        # Loop through voxel and place voxel time courses that are within the
        # mask in new hdf5 file:
        for idxVox in range(varNumVox):
            if vecLgcMsk[idxVox]:
                objQ.put(dtsFunc[:, idxVox])

        # Close thread:
        objThrd.join()

        # Close hdf5 files:
        fleHdf5.close()
        fleHdf5Msk.close()

        # Remove un-maksed (i.e. large) hdf5 file.
        os.remove(strPthHdf5)

        # ---------------------------------------------------------------------
        # Temporal smoothing

        # Read & write file (after masking):
        fleHdf5Msk = h5py.File(strPthHdf5Msk, 'r+')

        # Access dataset in current hdf5 file:
        dtsFunc = fleHdf5Msk['func']

        if 0.0 < varSdSmthTmp:

            print('---------Temporal smoothing')

            # Looping voxel by voxel is too slow. Instead, read & write a
            # chunks of voxels at a time. Indices of chunks:
            varStpSze = 100
            vecSplt = np.arange(0, (varNumVox + 1), varStpSze)
            vecSplt = np.concatenate((vecSplt, np.array([varNumVox])))

            # Number of chunks:
            varNumCnk = vecSplt.shape[0]

            # Buffer size:
            varBuff = 10

            # Create FIFO queue:
            objQ = queue.Queue(maxsize=varBuff)

            # Define & run extra thread with graph that places data on queue:
            objThrd = threading.Thread(target=feed_hdf5_spt,
                                       args=(dtsFunc, objQ, vecSplt))
            objThrd.setDaemon(True)
            objThrd.start()

            # Loop through chunks of volumes:
            for idxChnk in range((varNumCnk - 1)):

                # Start index of current chunk:
                varIdx01 = vecSplt[idxChnk]

                # Stop index of current chunk:
                varIdx02 = vecSplt[idxChnk + 1]

                # Get chunk of functional data from hdf5 file:
                aryFunc = dtsFunc[:, varIdx01:varIdx02]

                # Perform temporal smoothing:
                aryFunc = funcSmthTmp(0, aryFunc, varSdSmthTmp, None)

                # Put result on queue (from where it will be saved to disk in a
                # separate thread).
                objQ.put(aryFunc)

            # Close thread:
            objThrd.join()

        # ---------------------------------------------------------------------
        # Z-scoring

        print('---------Z-score functional data')

        # Looping voxel by voxel is too slow. Instead, read & write a chunks of
        # voxels at a time. Indices of chunks:
        varStpSze = 100
        vecSplt = np.arange(0, (varNumVoxMsk + 1), varStpSze)
        vecSplt = np.concatenate((vecSplt, np.array([varNumVoxMsk])))

        # Number of chunks:
        varNumCnk = vecSplt.shape[0]

        # Buffer size:
        varBuff = 10

        # Create FIFO queue:
        objQ = queue.Queue(maxsize=varBuff)

        # Define & run extra thread with graph that places data on queue:
        objThrd = threading.Thread(target=feed_hdf5_spt,
                                   args=(dtsFunc, objQ, vecSplt))
        objThrd.setDaemon(True)
        objThrd.start()

        # Loop through chunks of volumes:
        for idxChnk in range((varNumCnk - 1)):

            # Start index of current chunk:
            varIdx01 = vecSplt[idxChnk]

            # Stop index of current chunk:
            varIdx02 = vecSplt[idxChnk + 1]

            # Get chunk of functional data from hdf5 file:
            aryFunc = dtsFunc[:, varIdx01:varIdx02]

            # De-mean functional data:
            aryFunc = np.subtract(aryFunc,
                                  np.mean(aryFunc,
                                          axis=0,
                                          dtype=np.float32)[None, :])

            # Convert intensities into z-scores. If there are several pRF runs,
            # these are concatenated. Z-scoring ensures that differences in
            # mean image intensity and/or variance between runs do not confound
            # the analysis. Possible enhancement: Explicitly model across-runs
            # variance with a nuisance regressor in the GLM.
            aryTmpStd = np.std(aryFunc, axis=0)

            # In order to avoid devision by zero, only divide those voxels with
            # a standard deviation greater than zero:
            aryTmpLgc = np.greater(aryTmpStd.astype(np.float32),
                                   np.array([0.0], dtype=np.float32)[0])
            # Z-scoring:
            aryFunc[:, aryTmpLgc] = np.divide(aryFunc[:, aryTmpLgc],
                                              aryTmpStd[None, aryTmpLgc])
            # Set voxels with a variance of zero to intensity zero:
            aryTmpLgc = np.not_equal(aryTmpLgc, True)
            aryFunc[:, aryTmpLgc] = np.array([0.0], dtype=np.float32)[0]

            # Put result on queue (from where it will be saved to disk in a
            # separate thread).
            objQ.put(aryFunc)

        # Close thread:
        objThrd.join()

        # Close hdf5 file:
        fleHdf5Msk.close()

    # -------------------------------------------------------------------------
    # Combine runs

    # Path for hdf5 file with combined functional data from all runs (after
    # application of anatomical mask).
    strPthHdf5Conc = os.path.join(strFlePth, ('concat.hdf5'))

    # Create hdf5 file:
    fleHdf5Conc = h5py.File(strPthHdf5Conc, 'w')

    # Create dataset within hdf5 file:
    dtsFuncConc = fleHdf5Conc.create_dataset('func',
                                             (varNumVolTtl, varNumVoxMsk),
                                             dtype=np.float32)

    # Count volumes:
    varCntVol = 0

    # Loop through runs and load data:
    for idxRun in range(varNumRun):

        # Read hdf5 file (masked timecourses of current run):
        fleHdf5Msk = h5py.File(lstFleMsk[idxRun], 'r')

        # Access dataset in current hdf5 file:
        dtsFuncMsk = fleHdf5Msk['func']

        # Volumes in current run:
        varNumVolTmp = dtsFuncMsk.shape[0]

        # Looping volume by volume is too slow. Instead, read & write a chunk
        # of volumes at a time. Indices of chunks:
        varStpSze = 50
        vecSplt = np.arange(0, (varNumVolTmp + 1), varStpSze)
        vecSplt = np.concatenate((vecSplt, np.array([varNumVolTmp])))

        # Number of chunks:
        varNumCnk = vecSplt.shape[0]

        # Buffer size:
        varBuff = 10

        # Create FIFO queue:
        objQ = queue.Queue(maxsize=varBuff)

        # Account for previous runs:
        vecSpltPlus = np.add(vecSplt, varCntVol)

        # Define & run extra thread with graph that places data on queue:
        objThrd = threading.Thread(target=feed_hdf5_tme,
                                   args=(dtsFuncConc, objQ, vecSpltPlus))
        objThrd.setDaemon(True)
        objThrd.start()

        # Loop through chunks of volumes:
        for idxChnk in range((varNumCnk - 1)):

            # Start index of current chunk:
            varIdx01 = vecSplt[idxChnk]

            # Stop index of current chunk:
            varIdx02 = vecSplt[idxChnk + 1]

            # Number of volumes in current chunk:
            # varNumVolTmp = varIdx02 - varIdx01

            # Put current volumes on queue.
            objQ.put(dtsFuncMsk[varIdx01:varIdx02, :])

        # Close thread:
        objThrd.join()

        # Close hdf5 file (masked single run):
        fleHdf5Msk.close()

        # Remove maksed hdf5 file.
        os.remove(lstFleMsk[idxRun])

        # Increment volume counter:
        varCntVol += varNumVolTmp

    # Close hdf5 file (combined multi run):
    fleHdf5Conc.close()

    # -------------------------------------------------------------------------
    # Variance mask

    # Voxels that are outside the brain and have no, or very little, signal
    # should not be included in the pRF model finding. We take the variance
    # over time and exclude voxels with a suspiciously low variance.

    # Read concatenated hdf5 file:
    fleHdf5Conc = h5py.File(strPthHdf5Conc, 'r')

    # Access dataset in concatenated hdf5 file:
    dtsFuncConc = fleHdf5Conc['func']

    # Total number of volumes:
    varNumVolTtl = dtsFuncConc.shape[0]

    # Number of voxels:
    varNumVoxMsk = dtsFuncConc.shape[1]

    # Counter for voxels with variance greater than zero:
    varCntVoxInc = 0

    # Looping voxel by voxel is too slow. Instead, read & write a chunks of
    # voxels at a time. Indices of chunks:
    varStpSze = 100
    vecSplt = np.arange(0, (varNumVoxMsk + 1), varStpSze)
    vecSplt = np.concatenate((vecSplt, np.array([varNumVoxMsk])))

    # Number of chunks:
    varNumCnk = vecSplt.shape[0]

    # List for variance mask:
    lstLgcVar = [None] * (varNumCnk - 1)

    # Loop through chunks of voxels:
    for idxChnk in range((varNumCnk - 1)):

        # Start index of current chunk:
        varIdx01 = vecSplt[idxChnk]

        # Stop index of current chunk:
        varIdx02 = vecSplt[idxChnk + 1]

        # Get chunk of functional data from hdf5 file:
        aryFunc = dtsFuncConc[:, varIdx01:varIdx02]

        # Variance over time:
        vecVar = np.var(aryFunc, axis=0, dtype=np.float32)

        # Is the variance greater than zero?
        vecLgcVar = np.greater(vecVar,
                               np.array([0.0001]).astype(np.float32)[0])

        # Count voxels with variance greater than zero:
        varCntVoxInc += np.sum(vecLgcVar)

        # Remember variance mask:
        lstLgcVar[idxChnk] = np.copy(vecLgcVar)

    # Path for hdf5 file with combined functional data from all runs (after
    # application of variance mask).
    strPthHdf5Var = os.path.join(strFlePth, ('complete.hdf5'))

    strPthHdf5Func = strPthHdf5Var

    # Create hdf5 file:
    fleHdf5Var = h5py.File(strPthHdf5Var, 'w')

    # Create dataset within hdf5 file:
    dtsFuncVar = fleHdf5Var.create_dataset('func',
                                           (varNumVolTtl, varCntVoxInc),
                                           dtype=np.float32)

    # Index for placement of selected voxels (i.e. after application of
    # variance mask).
    varIdx03 = 0

    # Loop through chunks of voxels:
    for idxChnk in range((varNumCnk - 1)):

        # Start index of current chunk:
        varIdx01 = vecSplt[idxChnk]

        # Stop index of current chunk:
        varIdx02 = vecSplt[idxChnk + 1]

        # Get chunk of functional data from hdf5 file:
        aryFunc = dtsFuncConc[:, varIdx01:varIdx02]

        # Apply variance mask:
        aryFunc = aryFunc[:, lstLgcVar[idxChnk]]

        # Number of voxels in variance mask:
        varNumTmp = np.sum(lstLgcVar[idxChnk])

        # Place variance-masked data in hdf5 file:
        dtsFuncVar[:, varIdx03:(varIdx03 + varNumTmp)] = aryFunc

        # Increment counter:
        varIdx03 += varNumTmp

    # Close hdf5 files:
    fleHdf5Conc.close()
    fleHdf5Var.close()

    # Concatenate variance masks over all voxels, new shape:
    # `vecLgcVar[voxels]`.
    vecLgcVar = np.concatenate(lstLgcVar, axis=0)
    del(lstLgcVar)

    return vecLgcMsk, hdrMsk, aryAff, vecLgcVar, tplHdf5Shp, strPthHdf5Func


def pre_pro_models_hdf5(strPathMdl, varSdSmthTmp=2.0, varPar=10):
    """
    Preprocess pRF model time courses - hdf5 mode.

    Parameters
    ----------
    strPathMdl : str
        Path of file with pRF time course models (without file extension). In
        hdf5 mode, time courses are loaded to & saved to hdf5 file, so that
        not all pRF model time courses do not have to be loaded into RAM at
        once.
    varSdSmthTmp : float
        Extent of temporal smoothing that is applied to functional data and
        pRF time course models, [SD of Gaussian kernel, in seconds]. If `zero`,
        no temporal smoothing is applied.
    varPar : int
        Number of processes to run in parallel (multiprocessing).


    Returns
    -------
    This function has no return value. pRF model time courses are processed on
    disk (hdf5 mode).

    Notes
    -----
    pRF model time courses (in hdf5 file) have same shape as in 'regular' mode:
    aryPrfTc[x-position, y-position, SD, condition, volume].

    """
    print('------Preprocess pRF time course models (hdf5 mode).')

    # Path of hdf5 file:
    strPthHdf5 = (strPathMdl + '.hdf5')

    # Read file:
    fleHdf5 = h5py.File(strPthHdf5, 'r+')

    # Access dataset in current hdf5 file:
    aryPrfTc = fleHdf5['pRF_time_courses']

    # Loop through stimulus conditions, because the array needs to the 4D,
    # with time as last dimension, for the preprocessing. Otherwise the
    # same functions could not be used for the functional data and model
    # time courses (which would increase redundancy).
    varNumCon = aryPrfTc.shape[3]
    for idxCon in range(varNumCon):

        # Preprocessing of pRF time course models:
        aryPrfTc[:, :, :, idxCon, :] = pre_pro_par(
            aryPrfTc[:, :, :, idxCon, :], aryMask=np.array([]),
            lgcLinTrnd=False, varSdSmthTmp=varSdSmthTmp, varSdSmthSpt=0.0,
            varPar=varPar)

    # Close hdf5 file:
    fleHdf5.close()
