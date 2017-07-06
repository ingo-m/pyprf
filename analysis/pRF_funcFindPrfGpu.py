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


def funcFindPrfGpu(idxPrc, varNumX, varNumY, varNumPrfSizes, vecMdlXpos,  #noqa
                   vecMdlYpos, vecMdlSd, aryFunc, aryPrfTc, strVersion,
                   queOut):
    """
    Find the best pRF model for voxel time course.
    
    This version uses ```feed_dict``` to put model time courses on the
    computational graph. This is slow, i.e. slower than multi-threaded cython
    on CPU or GPU vesion using a queue.
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

    #    # Reshape pRF model time courses:
    #    aryPrfTc = np.reshape(aryPrfTc,
    #                          ((aryPrfTc.shape[0]
    #                            * aryPrfTc.shape[1]
    #                            * aryPrfTc.shape[2]),
    #                           aryPrfTc.shape[3]))
    
    #    # Reshape back to original shape:
    #    aryPrfTc = np.reshape(aryPrfTc,
    #                          (cfg.varNumX,
    #                           cfg.varNumY,
    #                           cfg.varNumPrfSizes,
    #                           cfg.varNumVol))

    # Constant term for the model:
    vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFunc = aryFunc.astype(np.float32)
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

    # There can be pRF model time courses with a variance of zero (i.e. pRF
    # models that are not actually responsive to the stimuli). For time
    # efficiency, and in order to avoid division by zero, we ignore these
    # model time courses.
    aryPrfTcVar = np.var(aryPrfTc, axis=3)

    # Zero with float32 precision for comparison:
    varZero32 = np.array(([0.0])).astype(np.float32)[0]

    # L2 regularization factor for regression:
    varL2reg = 0.0

    # Definition of computational graph:
    objGrph = tf.Graph()
    with objGrph.as_default():
        # Design matrix with two columns (graph input). The design matrix is
        # different on every iteration, so we define a placeholder object.
        objDsng = tf.placeholder(tf.float32, shape=(varNumVol, 2))  # ! 

        # Functional data. Because the functional data does not change, we
        # put the entire data on the graph. This may become a problem for
        # large datasets.
        objFunc = tf.Variable(aryFunc)

        # The matrix solving operation.
        # objMatSlve = tf.matrix_solve_ls(objDsng, objFunc, varL2reg, fast=True)

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

    # Create session with graph:
    with tf.Session(graph=objGrph) as objSess:

        # Initialise variables.
        tf.global_variables_initializer().run()

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
                            strStsMsg = ('------------Progress: ' +
                                         str(vecStatPrc[varCntSts01]) +
                                         ' % --- ' +
                                         str(vecStatPrf[varCntSts01]) +
                                         ' pRF models out of ' +
                                         str(varNumMdls))

                            print(strStsMsg)

                            # Only increment counter if the last value has not
                            # been reached yet:
                            if varCntSts01 < varStsStpSze:
                                varCntSts01 = varCntSts01 + int(1)
    
                    # Only fit pRF model if variance is not zero:
                    if np.greater(aryPrfTcVar[idxX, idxY, idxSd], varZero32):

                        # Current pRF time course model:
                        vecMdlTc = aryPrfTc[idxX, idxY, idxSd, :].flatten()
    
                        # We create a design matrix including the current pRF
                        # time course model, and a constant term:
                        aryDsgn = np.vstack([vecMdlTc,
                                             vecConst]).T
    
                        # Change type to float32:
                        aryDsgn = aryDsgn.astype(np.float32)
    
                        # Design matrix to nested list:
                        lstDsng = aryDsgn.tolist()

                        # Run the graph with current design matrix, returning
                        # parameter estimates (betas):
                        vecTmpRes = objSess.run(objMatSlve,
                                                feed_dict={objDsng:lstDsng})

                        #print(type(aryTmpCoef))
                        #print(aryTmpCoef.shape)

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
    vecFuncMean = np.mean(aryFunc, axis=0)
    # Deviation from the mean for each datapoint:
    vecFuncDev = np.subtract(aryFunc, vecFuncMean[None, :])
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
