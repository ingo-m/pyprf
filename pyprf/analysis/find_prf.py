# -*- coding: utf-8 -*-
"""Parent function for pRF finding."""

import numpy as np
import multiprocessing as mp
from pyprf.analysis.utilities import cls_set_config


def find_prf(dicCnfg, aryFunc, aryPrfTc=None, aryLgcMdlVar=None,
             strPrfTc=None):
    """
    Parent function for pRF finding.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing config parameters.
    aryFunc : np.array
        2D array with functional MRI data, with shape aryFunc[time, voxel].
    aryPrfTc : np.array or None
        Array with preprocessed pRF time course models, shape:
        aryPrfTc[x-position, y-position, SD, condition, volume]. `None` if in
        hdf5 mode (in that case, model time courses will be loaded from disk).
    aryLgcMdlVar : np.array or None
        Mask for pRF time courses with temporal variance greater than zero
        (i.e. models that are responsive to the stimulus). Can be used to
        restricted to models with a variance greater than zero. Shape:
        `aryLgcMdlVar[model-x-pos, model-y-pos, pRF-size]`. Only used if in
        hdf5 mode.
    strPrfTc : str or None
        Path of hdf5 file with preprocessed model time courses. Only used if in
        hdf5 mode.

    Returns
    -------
    lstPrfRes : list
        List with results of pRF finding, containing:
        idxPrc : int
            Process ID of the process calling this function (for CPU
            multi-threading). In GPU version, this parameter is 0.
        vecBstXpos : np.array
            1D array with best fitting x-position for each voxel, with shape
            vecBstXpos[voxel].
        vecBstYpos : np.array
            1D array with best fitting y-position for each voxel, with shape
            vecBstYpos[voxel].
        vecBstSd : np.array
            1D array with best fitting pRF size for each voxel, with shape
            vecBstSd[voxel].
        vecBstR2 : np.array
            1D array with R2 value of 'winning' pRF model for each voxel, with
            shape vecBstR2[voxel].

    Notes
    -----
    Parent function for pRF finding. This function calls one of several
    possible child functions to perform the actual pRF finding. There are
    options for pRF finding on CPU with cython or numpy, or on GPU. Also,
    there is an hdf5 mode (in case of large amounts of data, pRF models are not
    loaded into RAM but read from disk).

    """
    print('------Find pRF models for voxel time courses')

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # Conditional imports:
    if cfg.strVersion == 'gpu':
        from pyprf.analysis.find_prf_gpu import find_prf_gpu
    if ((cfg.strVersion == 'cython') or (cfg.strVersion == 'numpy')):
        from pyprf.analysis.find_prf_cpu import find_prf_cpu
        from pyprf.analysis.find_prf_cpu_hdf5 import find_prf_cpu_hdf5

    # Number of voxels for which pRF finding will be performed:
    varNumVoxInc = aryFunc.shape[1]

    print('---------Number of voxels on which pRF finding will be performed: '
          + str(varNumVoxInc))

    print('---------Preparing parallel pRF model finding')

    # For the GPU version, we need to set down the parallelisation to 1 now,
    # because no separate CPU threads are to be created. We may still use CPU
    # parallelisation for preprocessing, which is why the parallelisation
    # factor is only reduced now, not earlier.
    if cfg.strVersion == 'gpu':
        cfg.varPar = 1

    # Vector with the moddeled x-positions of the pRFs:
    vecMdlXpos = np.linspace(cfg.varExtXmin,
                             cfg.varExtXmax,
                             cfg.varNumX,
                             endpoint=True,
                             dtype=np.float32)

    # Vector with the moddeled y-positions of the pRFs:
    vecMdlYpos = np.linspace(cfg.varExtYmin,
                             cfg.varExtYmax,
                             cfg.varNumY,
                             endpoint=True,
                             dtype=np.float32)

    # Vector with the moddeled standard deviations of the pRFs:
    vecMdlSd = np.linspace(cfg.varPrfStdMin,
                           cfg.varPrfStdMax,
                           cfg.varNumPrfSizes,
                           endpoint=True,
                           dtype=np.float32)

    # Empty list for results (parameters of best fitting pRF model):
    lstPrfRes = [None] * cfg.varPar

    # Empty list for processes:
    lstPrcs = [None] * cfg.varPar

    # Create a queue to put the results in:
    queOut = mp.Queue()

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

    # Make sure type is float32:
    aryFunc = aryFunc.astype(np.float32)

    # In hdf5-mode, pRF time courses models are not loaded into RAM but
    # accessed from hdf5 file.
    if not(aryPrfTc is None):
        aryPrfTc = aryPrfTc.astype(np.float32)

    # Put functional data into chunks:
    for idxChnk in range(cfg.varPar):
        # Index of first voxel to be included in current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last voxel to be included in current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstFunc[idxChnk] = aryFunc[:, varTmpChnkSrt:varTmpChnkEnd]

    # We don't need the original array with the functional data anymore:
    del(aryFunc)

    # CPU version (using numpy or cython for pRF finding):
    if ((cfg.strVersion == 'numpy') or (cfg.strVersion == 'cython')):

        print('---------pRF finding on CPU')

        print('---------Creating parallel processes')

        # Create processes:
        for idxPrc in range(cfg.varPar):

            # Hdf5-mode?
            if aryPrfTc is None:

                # Hdf5-mode (access pRF model time courses from disk in order
                # to avoid out of memory).
                lstPrcs[idxPrc] = mp.Process(target=find_prf_cpu_hdf5,
                                             args=(idxPrc,
                                                   vecMdlXpos,
                                                   vecMdlYpos,
                                                   vecMdlSd,
                                                   lstFunc[idxPrc],
                                                   strPrfTc,
                                                   aryLgcMdlVar,
                                                   cfg.strVersion,
                                                   queOut)
                                             )

            else:

                # Regualar CPU mode.
                lstPrcs[idxPrc] = mp.Process(target=find_prf_cpu,
                                             args=(idxPrc,
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

    # GPU version (using tensorflow for pRF finding):
    elif cfg.strVersion == 'gpu':

        # The following features are currently not available in GPU mode:
        # - Handling of multiple predictors (e.g. contrast levels).
        # - Export of parameter estimates.
        # - Hdf5 mode.

        # Assert that hdf5 mode has not been requested.
        strMsg = ('Hdf5 mode not implemented for GPU mode.')
        # assert not(aryPrfTc is None), strMsg
        assert not(cfg.lgcHdf5), strMsg

        # Assert that there is only one contrast level.
        strMsg = ('Handling of multiple predictors (e.g. contrast levels) not '
                  + 'implemented for GPU mode (switch to numpy or cython '
                  + 'mode.')
        assert (aryPrfTc.shape[3] == 1), strMsg

        # Reshape:
        aryPrfTc = aryPrfTc[:, :, :, 0, :]

        print('---------pRF finding on GPU')

        # Create processes:
        for idxPrc in range(cfg.varPar):

            # Hdf5-mode (access pRF model time courses from disk in order
            # to avoid out of memory).
            lstPrcs[idxPrc] = mp.Process(target=find_prf_gpu,
                                         args=(idxPrc,
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
    for idxPrc in range(cfg.varPar):
        lstPrcs[idxPrc].start()

    # Delete reference to list with function data (the data continues to exists
    # in child process):
    del(lstFunc)

    # Collect results from queue:
    for idxPrc in range(cfg.varPar):
        lstPrfRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(cfg.varPar):
        lstPrcs[idxPrc].join()

    return lstPrfRes
