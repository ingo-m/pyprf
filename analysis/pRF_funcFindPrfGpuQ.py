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

# Number of x-positions to model:
varNumX = 40
# Number of y-positions to model:
varNumY = 40
# Number of pRF sizes to model:
varNumPrfSizes = 40


def funcFindPrfGpu(idxPrc, varNumX, varNumY, varNumPrfSizes, vecMdlXpos,  #noqa
                   vecMdlYpos, vecMdlSd, aryFunc, aryPrfTc, queOut):
    """
    Find the best pRF model for voxel time course.
    
    Testing version for queues.
    """
    # -------------------------------------------------------------------------
    # *** Wueue-feeding-function that will run in extra thread
    def funcPlcIn():
        """Function for placing data on queue."""

        # Iteration counter:
        idxCnt = 0

        while True:

            # Feed example to Tensorflow placeholder
            dicIn = {objPlcHld01: lstPrfTc[idxCnt]}

            # Push to the queue:
            objSess.run(objEnQ, feed_dict=dicIn)

            idxCnt += 1

            # Stop if coordinator says stop:
            if objCoord.should_stop():
                break

            # Stop if all data has been put on the queue:        
            elif idxCnt == varNumMdl:
                break

    # -------------------------------------------------------------------------
    # *** Prepare input data

    print('------Prepare input data')

    # Number of voxels to be fitted:
    varNumVoxChnk = aryFunc.shape[0]

    # Number of volumes:
    varNumVol = aryFunc.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000000.0).astype(np.float32)

    # Vector that will hold the temporary residuals from the model fitting:
    vecTmpRes = np.zeros(varNumVoxChnk).astype(np.float32)

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

    # L2 regularization factor for regression:
    varL2reg = 0.0

    # Put pRF model time courses into list:
    lstPrfTc = [None] * aryPrfTc.shape[0]
    for idxMdl in range(int(aryPrfTc.shape[0])):
        lstPrfTc[idxMdl] = aryPrfTc[idxMdl, :, :]
    del(aryPrfTc)

    # Total number of pRF models to fit:
    varNumMdl = len(lstPrfTc)


    # -------------------------------------------------------------------------
    # *** Define computational graph, queue & session

    print('------Define computational graph, queue & session')

    # Queue capacity:
    varCapQ = 20

    # The queue:
    objQ = tf.FIFOQueue(capacity=varCapQ, dtypes=[tf.float32])

    # Method for getting queue size:
    objSzeQ = objQ.size()

    # Placeholder that is used to put design matrix on computational graph:
    objPlcHld01 = tf.placeholder(tf.float32,
                                 shape=list((lstPrfTc[0].shape[0],
                                             lstPrfTc[0].shape[1])))

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

    # Functional data. Because the functional data does not change, we put the
    # entire data on the graph. This may become a problem for large datasets.
    objFunc = tf.Variable(aryFunc)

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

    # Define session:
    objSess = tf.Session()

    # Coordinator needs to be initialised as well:
    objCoord = tf.train.Coordinator()

    # -------------------------------------------------------------------------
    # *** Fill queue

    print('------Fill queue')

    # Buffer size (number of samples to put on queue before starting execution
    # of graph):
    varBuff = 20

    # Define & run extra thread with graph that places data on queue:
    objThrd = threading.Thread(target=funcPlcIn)
    objThrd.setDaemon(True)
    objThrd.start()

    # Stay in this while loop until the specified number of samples (varBuffer)
    # have been placed on the queue).
    varTmpSzeQ = 0
    while varTmpSzeQ < varBuff:
        varTmpSzeQ = objSess.run(objSzeQ)

    # -------------------------------------------------------------------------
    # *** Run the graph

    print('------Run graph')

    # Variables need to be initialised:
    objSess.run(tf.global_variables_initializer())

    # List for results:
    lstRes = [None] * varNumMdl

    # Loop through input iterations:
    for idxIt in range(varNumMdl):

        # Run main computational graph and put results in list:
        lstRes[idxIt] = objSess.run(objMatSlve)

        # On every 1000th call, check number of elements on queue:
        if (idxIt % 1000) == 0:

            # Number of elements on queue:
            varTmpSzeQ = objSess.run(objSzeQ)
    
            strTmpMsg = ('---------Iteration: '
                         + str(idxIt)
                         + ', number of elements on queue: '
                         + str(varTmpSzeQ))

            print(strTmpMsg)

    print(type(lstRes))
    print(len(lstRes))

    print(type(lstRes[0]))
    print(lstRes[0].shape)

    # Stop threads.
    objCoord.request_stop()
    objSess.close()

    # -------------------------------------------------------------------------


# TODO

# Put residuals from list into array, find model with lowest residuals along
# model-dimension, and output respective model parameters



#    # Check whether current residuals are lower than previously
#    # calculated ones:
#    vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)
#
#    # Replace best x and y position values, and SD values.
#    vecBstXpos[vecLgcTmpRes] = vecMdlXpos[idxX]
#    vecBstYpos[vecLgcTmpRes] = vecMdlYpos[idxY]
#    vecBstSd[vecLgcTmpRes] = vecMdlSd[idxSd]
#
#    # Replace best residual values:
#    vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]
#
#    # After finding the best fitting model for each voxel, we still have to
#    # calculate the coefficient of determination (R-squared) for each voxel. We
#    # start by calculating the total sum of squares (i.e. the deviation of the
#    # data from the mean). The mean of each time course:
#    vecFuncMean = np.mean(aryFunc, axis=0)
#    # Deviation from the mean for each datapoint:
#    vecFuncDev = np.subtract(aryFunc, vecFuncMean[None, :])
#    # Sum of squares:
#    vecSsTot = np.sum(np.power(vecFuncDev,
#                               2.0),
#                      axis=0)
#    # Coefficient of determination:
#    vecBstR2 = np.subtract(1.0,
#                           np.divide(vecBstRes,
#                                     vecSsTot))
#
#    # Output list:
#    lstOut = [idxPrc,
#              vecBstXpos,
#              vecBstYpos,
#              vecBstSd,
#              vecBstR2]

#lstOut = ['error']

#queOut.put(lstOut)

