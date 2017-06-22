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


def funcFindPrfGpu(varNumX, varNumY, varNumPrfSizes, vecMdlXpos,  #noqa
                   vecMdlYpos, vecMdlSd, aryFunc, aryPrfTc):
    """Find the best pRF model for voxel time course."""
    # Number of voxels to be fitted:
    varNumVoxChnk = aryFunc.shape[0]

    # Number of volumes:
    varNumVol = aryFunc.shape[1]

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
    aryFunc = aryFunc.T

    # Constant term for the model:
    vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFunc = aryFunc.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

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

    # "l2_regularizer: 0-D double Tensor":
    varL2reg = 1e6 # L2 regularization factor



    aryFunc = aryFunc[:, 0:10000]


    # define the computational graph
    graph = tf.Graph()
    with graph.as_default():
        # declare graph inputs
        x_train = tf.placeholder(tf.float32, shape=(varNumVol, 2))  # ! Design matrix with two columns
        #y_train = tf.placeholder(tf.float32, shape=(varNumVol, varNumVoxChnk)) # ! Data

        y_train = tf.Variable(aryFunc)

        theta = tf.Variable([[0.0], [0.0]]) # implicit bias!  # ! Initial values?
        # optimum
        #optimum = tf.matrix_solve_ls(x_train, y_train, LAMBDA, fast=True)
        optimum = tf.matrix_solve_ls(x_train, y_train, varL2reg, fast=True)


    # Put functional data into nested list:
    #lstFunc = aryFunc.tolist()

    #print('print(len(lstFunc))')
    #print(len(lstFunc))

    #print('print(len(lstFunc[0]))')
    #print(len(lstFunc[0]))


    #del(aryFunc)

    #lalala = tf.Variable(np.ones((2,2)))


    # run the computation: no loop needed!
    with tf.Session(graph=graph) as s:
        tf.initialize_all_variables().run()




        # Loop through pRF models:
        for idxX in range(0, varNumX):
    
            for idxY in range(0, varNumY):
    
                for idxSd in range(0, varNumPrfSizes):
    
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
    
                        # Only increment counter if the last value has not been
                        # reached yet:
                        if varCntSts01 < varStsStpSze:
                            varCntSts01 = varCntSts01 + int(1)
    
                    # Only fit pRF model if variance is not zero:
                    if np.greater(aryPrfTcVar[idxX, idxY, idxSd], varZero32):
    
    
    
                        # Numpy version:
    
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


                        #opt = s.run(optimum, feed_dict={x_train:lstDsng, y_train:lstFunc})
                        opt = s.run(optimum, feed_dict={x_train:lstDsng})


                    # Calculate the least-squares solution for all voxels:
                    #vecTmpRes = np.linalg.lstsq(aryDsgn, aryFunc)[1]



#                # Check whether current residuals are lower than previously
#                # calculated ones:
#                vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)
#
#                # Replace best x and y position values, and SD values.
#                vecBstXpos[vecLgcTmpRes] = vecMdlXpos[idxX]
#                vecBstYpos[vecLgcTmpRes] = vecMdlYpos[idxY]
#                vecBstSd[vecLgcTmpRes] = vecMdlSd[idxSd]
#
#                # Replace best residual values:
#                vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]
#
#                # Increment status indicator counter:
#                varCntSts02 = varCntSts02 + 1
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
#    lstOut = [vecBstXpos,
#              vecBstYpos,
#              vecBstSd,
#              vecBstR2]
#
#    return lstOut
