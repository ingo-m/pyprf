# -*- coding: utf-8 -*-
"""pRF finding function definitions."""

# Part of py_pRF_mapping library
# Copyright (C) 2018  Ingo Marquardt
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
import h5py
import threading
import queue
import numpy as np
import nibabel as nb


def nii_to_hdf5(strPathIn):
    """
    Copy data from nii to hdf5 format.

    Parameters
    ----------
    strPathIn : str
        Path to nii file to save as hdf5.

    Returns
    -------
    This function has no return value.

    Notes
    -----
    In case large datasets, fMRI data may not fit into memory. Therefore, the
    time courses are copyed into hdf5 format, and read from disk during model
    finding. A flattened version of the fMRI time courses is saved to disk,
    with shape: func[time, voxel].

    """
    # File path & file name:
    strFlePth, strFleNme = os.path.split(strPathIn)

    # Remove file extension from file name:
    strFleNme = strFleNme.split('.')[0]

    # Prepare placement of pRF model time courses in hdf5 file.

    # Buffer size:
    varBuff = 10

    # Create FIFO queue:
    objQ = queue.Queue(maxsize=varBuff)

    # Path of hdf5 file:
    strPthHdf5 = os.path.join(strFlePth, (strFleNme + '.hdf5'))

    # Only create hdf5 file if it does not exist yet:
    if not(os.path.isfile(strPthHdf5)):

        print(('------------File: ' + strFleNme + '.hdf5'))

        # Create hdf5 file:
        fleHdf5 = h5py.File(strPthHdf5, 'w')

        # Load nii file (this does not load the data into memory yet):
        objNii = nb.load(strPathIn)

        # Shape of nii file:
        tplSze = objNii.shape

        # Number of voxels:
        varNumVox = (tplSze[0] * tplSze[1] * tplSze[2])

        # Number of volumes:
        varNumVol = tplSze[3]

        # Create dataset within hdf5 file:
        dtsFunc = fleHdf5.create_dataset('func',
                                         (varNumVol, varNumVox),
                                         dtype=np.float32)

        # Looping volume by volume is too slow. Instead, read & write a chunk
        # of volumes at a time. Indices of chunks:
        varStpSze = 50
        vecSplt = np.arange(0, (varNumVol + 1), varStpSze)

        # Concatenate stop index of last chunk (only if there are remaining
        # voxels after the last chunk).
        if not(vecSplt[-1] == varNumVol):
            vecSplt = np.concatenate((vecSplt, np.array([varNumVol])))

        # Number of chunks:
        varNumCnk = vecSplt.shape[0]

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
            varNumVolTmp = varIdx02 - varIdx01

            # Load from nii file, and reshape (new shape: func[time, voxel]).
            aryTmp = np.asarray(objNii.dataobj[..., varIdx01:varIdx02]
                                ).astype(np.float32)

            # The reshape mechanism need to be the same as in the 'regular'
            # (i.e. not hdf5 mode) pipeline (`pre_pro_func`).
            aryTmp = np.reshape(aryTmp, [varNumVox, varNumVolTmp]).T
            # aryTmp = np.reshape(aryTmp, [varNumVolTmp, varNumVox])

            # Put current volume on queue.
            objQ.put(aryTmp)

        # Close queue feeding thread, and hdf5 file.

        # Close thread:
        objThrd.join()

        # Close file:
        fleHdf5.close()


def feed_hdf5_tme(dtsFunc, objQ, vecSplt):
    """
    Feed FIFO queue for placement of voxel time courses segments in hdf5 file.

    Parameters
    ----------
    dtsFunc : h5py dataset
        Dataset within h5py file.
    objQ : queue.Queue
        Queue from which voxel time courses are retrieved.
    vecSplt : np.array
        Indices of chunks of volumes in fMRI time course. Looping volume by
        volume is too slow. Instread, read & write a chunk of volumes at a
        time.

    Notes
    -----
    Place temporal segments (i.e. chunks of volumes) in hdf5 file. There is a
    corresponding function for placement of spatial segments.

    """
    # Number of chunks:
    varNumCnk = vecSplt.shape[0]

    # Loop through chunks of volumes:
    for idxChnk in range((varNumCnk - 1)):

        # Start index of current chunk:
        varIdx01 = vecSplt[idxChnk]

        # Stop index of current chunk:
        varIdx02 = vecSplt[idxChnk + 1]

        # Take voxel time course from queue, and put in hdf5 file:
        dtsFunc[varIdx01:varIdx02, :] = objQ.get()


def feed_hdf5_spt(dtsFunc, objQ, vecSplt):
    """
    Feed FIFO queue for placement of spatial segments in hdf5 file.

    Parameters
    ----------
    dtsFunc : h5py dataset
        Dataset within h5py file.
    objQ : queue.Queue
        Queue from which voxel time courses are retrieved.
    vecSplt : np.array
        Indices of chunks of voxels in fMRI time course. Looping voxels by
        voxels is too slow. Instread, read & write a chunk of voxels at a
        time.

    Notes
    -----
    Place spatial segments (i.e. chunks of voxels) in hdf5 file. There is a
    corresponding function for placement of temporal segments.

    """
    # Number of chunks:
    varNumCnk = vecSplt.shape[0]

    # Loop through chunks of volumes:
    for idxChnk in range((varNumCnk - 1)):

        # Start index of current chunk:
        varIdx01 = vecSplt[idxChnk]

        # Stop index of current chunk:
        varIdx02 = vecSplt[idxChnk + 1]

        # Take voxel time course from queue, and put in hdf5 file:
        dtsFunc[:, varIdx01:varIdx02] = objQ.get()


def feed_hdf5(dtsFunc, objQ, varNumVox):
    """
    Feed FIFO queue for placement of single voxel time courses in hdf5 file.

    Parameters
    ----------
    dtsFunc : h5py dataset
        Dataset within h5py file.
    objQ : queue.Queue
        Queue from which voxel time courses are retrieved.
    varNumVox : int
        Number of voxels in target hdf5 file.

    Notes
    -----
    Place single voxel time courses in hdf5 file.

    """
    # Loop through voxels:
    for idxVox in range(varNumVox):

        # Take model time course from queue, and put in hdf5 file:
        dtsFunc[:, idxVox] = objQ.get()


#if __name__ == "__main__":
#
#    strPathIn = '/home/john/Desktop/tmp/ttt/sub-02_ses-01_run_01.nii.gz'
#
#    nii_to_hdf5(strPathIn)

#def read_hdf5_q(dtsFunc, objQ, varNumVol):
#    """
#    Read voxel time courses from hdf5 file & put on FIFO queue.
#
#    Parameters
#    ----------
#    dtsFunc : h5py dataset
#        Dataset within h5py file.
#    objQ : queue.Queue
#        Queue on which to place voxel time course.
#    tplCoor : tuple
#        Tuple with coordinates of voxel time course to read (for instance, if
#        `tplCoor = (1, 2, 3, :)`, the entire voxel time course with coordinates
#        x = 1, y = 2, z = 3 will be read.
#
#    """
#    # Loop through volumes:
#    for idxVol in range(varNumVol):
#
#            # Take voxel time course from queue, and put in hdf5 file:
#            dtsFunc[:, :, :, idxVol] = objQ.get()
