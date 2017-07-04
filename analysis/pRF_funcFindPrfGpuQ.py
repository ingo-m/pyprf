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

    lstPrfTc = [None] * aryPrfTc.shape[0]
    for idx01 in range(int(aryPrfTc.shape[0])):
        lstPrfTc[idx01] = aryPrfTc[idx01, :, :]
    del(aryPrfTc)

    print('------Define computational graph')

    varNumPar = 1

    #print('print(lstPrfTc[0].shape)')
    print(lstPrfTc[0].shape)

    # Define computational graph:
    objGrph = tf.Graph()
    with objGrph.as_default():
    #with tf.variable_scope("queue"):
        # Design matrix with two columns (graph input). The design matrix is
        # different on every iteration, so we define a placeholder object.
        # objDsng = tf.placeholder(tf.float32, shape=(varNumVol, 2))  # ! 

        objQueue = tf.FIFOQueue(capacity=1, dtypes=tf.float32)

        objEnqueue = objQueue.enqueue_many(lstPrfTc[0:50])

        objQRunner = tf.train.QueueRunner(objQueue, [objEnqueue] * varNumPar)

        tf.train.add_queue_runner(objQRunner)

        # Functional data. Because the functional data does not change, we
        # put the entire data on the graph. This may become a problem for
        # large datasets.
        objFunc = tf.Variable(aryFunc[:, 0:100])

        # The matrix solving operation.
        # objMatSlve = tf.matrix_solve_ls(objDsng, objFunc, varL2reg, fast=True)

        objDsng = objQueue.dequeue()

        # Operation that solves matrix (in the least squares sense), and
        # calculates residuals along time dimension:
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

    print('------Create session')

    # Create session with graph:
    with tf.Session(graph=objGrph) as objSess:
    #with tf.Session() as objSess:

        # Initialise variables.
        tf.global_variables_initializer().run()

        # ... add the coordinator, ...
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('------Run computational graph')

        # Run the graph with current design matrix, returning
        # parameter estimates (betas):
        vecTmpRes = objSess.run(objMatSlve)

        print(type(vecTmpRes))
        print(vecTmpRes.shape)

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

    lstOut = ['error']

    queOut.put(lstOut)

