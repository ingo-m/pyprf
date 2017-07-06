# -*- coding: utf-8 -*-
"""Main function for pRF finding."""

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

import numpy as np
import tensorflow as tf
import threading


def funcFindPrfGpu(idxPrc, varNumX, varNumY, varNumPrfSizes, vecMdlXpos,  #noqa
                   vecMdlYpos, vecMdlSd, aryFunc, aryPrfTc, queOut):
    """
    Find best pRF model for voxel time course.
    
    This version uses a queue that runs in a separate thread to put model time
    courses on the computational graph.
    """
    # -------------------------------------------------------------------------
    # *** Queue-feeding-function that will run in extra thread
    def funcPlcIn():
        """Function for placing data on queue."""

        # Iteration counter:
        idxCnt = 0

        while True:

            # Feed example to Tensorflow placeholder
            aryTmp02 = np.copy(lstPrfTc[idxCnt])
            dicIn = {objPlcHld01: aryTmp02}

            # Push to the queue:
            objSess.run(objEnQ, feed_dict=dicIn)

            idxCnt += 1

            # Stop if coordinator says stop:
            if objCoord.should_stop():
                break

            # Stop if all data has been put on the queue:        
            elif idxCnt == varNumMdls:
                break

    # -------------------------------------------------------------------------
    # *** Prepare input data

    print('------Prepare input data')

    # Number of voxels to be fitted:
    varNumVox = aryFunc.shape[0]

    # Number of volumes:
    # varNumVol = aryFunc.shape[1]

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFunc = aryFunc.T

    # Reshape pRF model time courses:
    aryPrfTc = np.reshape(aryPrfTc,
                          ((aryPrfTc.shape[0]
                            * aryPrfTc.shape[1]
                            * aryPrfTc.shape[2]),
                           aryPrfTc.shape[3]))

    # Change type to float 32:
    aryFunc = aryFunc.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # The pRF model is fitted only if variance along time dimension is not
    # zero. Get variance along time dimension:
    vecVarPrfTc = np.var(aryPrfTc, axis=1)

    # Zero with float32 precision for comparison:
    varZero32 = np.array(([0.0])).astype(np.float32)[0]

    # Boolean array for models with variance greater than zero:
    vecLgcVar = np.greater(vecVarPrfTc, varZero32)

    # Take models with variance less than zero out of the array:
    aryPrfTc = aryPrfTc[vecLgcVar, :]

    # Add extra dimension for constant term:
    aryPrfTc = np.reshape(aryPrfTc, (aryPrfTc.shape[0], aryPrfTc.shape[1], 1))

    # Add constant term (ones):
    aryPrfTc = np.concatenate((aryPrfTc,
                               np.ones(aryPrfTc.shape).astype(np.float32)),
                              axis=2)

    # Change type to float 32:
    aryFunc = aryFunc.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

    # Size of pRF time courses in MB:
    varSzePrf = np.divide(float(aryPrfTc.nbytes),
                           1000000.0)

    print(('---------Size of pRF time courses: '
           + str(np.around(varSzePrf))
           + ' MB'))

    # L2 regularization factor for regression:
    varL2reg = 0.0

    # Put pRF model time courses into list:
    lstPrfTc = [None] * aryPrfTc.shape[0]
    for idxMdl in range(int(aryPrfTc.shape[0])):
        lstPrfTc[idxMdl] = aryPrfTc[idxMdl, :, :]
    del(aryPrfTc)

    # Total number of pRF models to fit:
    varNumMdls = len(lstPrfTc)

    # We cannot commit the entire functional data to GPU memory, we need to
    # create chunks. Establish the limit (maximum size) of one chunk (in MB):
    varSzeMax = 200.0

    # Size of functional data in MB:
    varSzeFunc = np.divide(float(aryFunc.nbytes),
                           1000000.0)

    print(('---------Size of functional data: '
           + str(np.around(varSzeFunc))
           + ' MB'))

    # Number of chunks to create:
    varNumChnk = int(np.ceil(np.divide(varSzeFunc, varSzeMax)))

    print(('---------Functional data will be split into '
           + str(varNumChnk)
           + ' batches'))

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumVox,
                              num=varNumChnk,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumVox))

    # We delete the original array holding the functional data to conserve
    # memory. Therefore, we first need to calculate the mean (will be needed
    # for calculation of R2).

    # After finding the best fitting model for each voxel, we still have to
    # calculate the coefficient of determination (R-squared) for each voxel. We
    # start by calculating the total sum of squares (i.e. the deviation of the
    # data from the mean). The mean of each time course:
    vecFuncMean = np.mean(aryFunc, axis=0)
    # Deviation from the mean for each datapoint:
    vecFuncDev = np.subtract(aryFunc, vecFuncMean[None, :])
    # Sum of squares:
    vecSsTot = np.sum(np.power(vecFuncDev,
                               2.0),
                      axis=0)

    # List into which the chunks of functional data are put:
    lstFunc = [None] * varNumChnk

    # Put functional data into chunks:
    for idxChnk in range(0, varNumChnk):
        # Index of first voxel to be included in current chunk:
        varChnkStr = int(vecIdxChnks[idxChnk])
        # Index of last voxel to be included in current chunk:
        varChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstFunc[idxChnk] = aryFunc[:, varChnkStr:varChnkEnd]

    # We don't need the original array with the functional data anymore (the
    # above seems to have created a hard copy):
    del(aryFunc)

    # Vector for minimum residuals:
    vecResMin = np.zeros((varNumVox), dtype=np.float32)

    # -------------------------------------------------------------------------
    # *** Prepare status indicator

    # We create a status indicator for the time consuming pRF model finding
    # algorithm. Number of steps of the status indicator:
    varStsStpSze = 20

    # Vector with pRF values at which to give status feedback:
    vecStatPrf = np.linspace(0,
                             (varNumMdls * varNumChnk),
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

    # -------------------------------------------------------------------------
    # *** Define computational graph, queue & session

    print('------Define computational graph, queue & session')

    # Queue capacity:
    varCapQ = 100

    # Dimensions of placeholder have to be determined outside of the tensor
    # object, otherwise the object on which the size is calculated is loaded
    # into GPU memory.
    varDim01 = lstPrfTc[0].shape[0]
    varDim02 = lstPrfTc[0].shape[1]

    # The queue:
    objQ = tf.FIFOQueue(capacity=varCapQ,
                        dtypes=[tf.float32],
                        shapes=[(varDim01, varDim02)])

    # Method for getting queue size:
    objSzeQ = objQ.size()

    # Placeholder that is used to put design matrix on computational graph:
    objPlcHld01 = tf.placeholder(tf.float32,
                                 shape=[varDim01, varDim02])

    # The enqueue operation that puts data on the graph.
    objEnQ = objQ.enqueue([objPlcHld01])

    # Number of threads that will be created:
    varNumThrd = 1

    # The queue runner (places the enqueue operation on the queue?).
    objRunQ = tf.train.QueueRunner(objQ, [objEnQ] * varNumThrd)
    tf.train.add_queue_runner(objRunQ)

    # The tensor object that is retrieved from the queue. Functions like
    # placeholders for the data in the queue when defining the graph.
    objDsng = objQ.dequeue()

    # Functional data. Here we put the first chunk of data on the graph. This
    # needs to be updated later for subsequent chunks.
    aryTmp01 = np.copy(lstFunc[0])
    objFunc = tf.Variable(aryTmp01)

    # The computational graph. Operation that solves matrix (in the least
    # squares sense), and calculates residuals along time dimension:
    objMatSlve = tf.reduce_sum(
                               tf.abs(
                                      tf.subtract(
                                                  tf.matmul(
                                                            objDsng,
                                                            tf.matrix_solve_ls( \
                                                                objDsng, objFunc,
                                                                varL2reg,
                                                                fast=True)
                                                            ),
                                                  objFunc),
                                      ),
                               axis=0
                               )

    # -------------------------------------------------------------------------
    # *** Loop through chunks

    print('------Run graph')

    for idxChnk in range(varNumChnk):

        print(('---------Chunk: ' + str(idxChnk)))

        # Define session:
        objSess = tf.Session()

        # Coordinator needs to be initialised as well:
        objCoord = tf.train.Coordinator()

        # ---------------------------------------------------------------------
        # *** Fill queue

        # Buffer size (number of samples to put on queue before starting
        # execution of graph):
        varBuff = 100

        # Define & run extra thread with graph that places data on queue:
        objThrd = threading.Thread(target=funcPlcIn)
        objThrd.setDaemon(True)
        objThrd.start()

        # Stay in this while loop until the specified number of samples
        # (varBuffer) have been placed on the queue).
        varTmpSzeQ = 0
        while varTmpSzeQ < varBuff:
            varTmpSzeQ = objSess.run(objSzeQ)

        # ---------------------------------------------------------------------
        # *** Run the graph

        # One the second and all subsequent iterations of the loop, a new
        # chunk of functional data needs to be placed on the graph.
        if 0 < idxChnk:

            # New chunk of functional data:
            aryTmp01 = np.copy(lstFunc[idxChnk])
            objFunc = tf.Variable(aryTmp01)

        # Variables need to be (re-)initialised:
        objSess.run(tf.global_variables_initializer())

        # Index of first voxel in current chunk (needed to assign results):
        varChnkStr = int(vecIdxChnks[idxChnk])

        # Index of last voxel in current chunk (needed to assign results):
        varChnkEnd = int(vecIdxChnks[(idxChnk+1)])

        # Array for results of current chunk:
        aryTmpRes = np.zeros((varNumMdls,
                              lstFunc[idxChnk].shape[1]),
                             dtype=np.float32)

        # Loop through models:
        for idxMdl in range(varNumMdls):

            # Run main computational graph and put results in list:
            aryTmpRes[idxMdl, :] = objSess.run(objMatSlve)

            # Status indicator:
            if varCntSts02 == vecStatPrf[varCntSts01]:
                # Number of elements on queue:
                varTmpSzeQ = objSess.run(objSzeQ)
                # Prepare status message:
                strStsMsg = ('---------Progress: '
                             + str(vecStatPrc[varCntSts01])
                             + ' % --- Number of elements on queue: '
                             + str(varTmpSzeQ))
                print(strStsMsg)
                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

        # Stop threads.
        objCoord.request_stop()
        objSess.close()

        # Get indices of models with minimum residuals (minimum along
        # model-space) for current chunk:
        vecResMin[varChnkStr:varChnkEnd] = np.argmin(aryTmpRes, axis=0)

    # -------------------------------------------------------------------------
    # *** Post-process results

    print('------Post-processing results')

    # Array for model parameters. At the moment, we have the indices of the
    # best fitting models, so we need an array that tells us what model
    # parameters these indices refer to.
    aryMdl = np.zeros((varNumMdls, 3), dtype=np.float32)

    # Model parameter can be represented as float32 as well:
    vecMdlXpos = vecMdlXpos.astype(np.float32)
    vecMdlYpos = vecMdlYpos.astype(np.float32)
    vecMdlSd = vecMdlSd.astype(np.float32)

    # The first column is to contain model x positions:
    aryMdl[:, 0] = np.repeat(vecMdlXpos, int(varNumY * varNumPrfSizes))

    # The second column is to contain model y positions:
    aryMdl[:, 1] = np.repeat(
                             np.tile(vecMdlYpos,
                                     varNumPrfSizes),
                             varNumX
                             )

    # The third column is to contain model pRF sizes:
    aryMdl[:, 2] = np.tile(vecMdlSd, int(varNumX * varNumY))

    # The above code has the same result as the below (for better readability):
    # aryMdl = np.zeros((varNumMdls, 3), dtype=np.float32)
    # varCount = 0
    # # Loop through pRF models:
    # for idxX in range(0, varNumX):
    #     for idxY in range(0, varNumY):
    #         for idxSd in range(0, varNumPrfSizes):
    #             aryMdl[varCount, 0] = vecMdlXpos[idxX]
    #             aryMdl[varCount, 1] = vecMdlYpos[idxY]
    #             aryMdl[varCount, 2] = vecMdlSd[idxSd]
    #             varCount += 1

    # Earlier, we had removed models with a variance of less than zero. Thus
    # those models were ignored and are not present in the results. We remove
    # them from the model-parameter-array:
    aryMdl = aryMdl[vecLgcVar]

    # Retrieve model parameters of 'winning' model for all voxels:
    vecBstXpos = aryMdl[:, 0][vecResMin]
    vecBstYpos = aryMdl[:, 1][vecResMin]
    vecBstSd = aryMdl[:, 2][vecResMin]

    # Coefficient of determination:
    vecBstR2 = np.subtract(1.0,
                           np.divide(vecResMin,
                                     vecSsTot))

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              vecBstR2]

    queOut.put(lstOut)
