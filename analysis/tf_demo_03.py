# -*- coding: utf-8 -*-
"""
Simple tensorflow demo using queue to place input data on graph.

This version uses a separate graph, running in a separate thread, to
place data on the queue. GLM fitting is tested.
"""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import time
import threading
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt


def funcGlmGpu(varNumVol=500, varNumChnk=2, varNumVoxPerChnk=200000,
               varNumBeta=2, varNumMdl=1000):
    """Simulate GLM fitting on GPU."""
    # Define queue-feeding-function that will run in extra thread:
    def funcPlcIn():
        """Place data on queue."""
        # Iteration counter:
        varCntIt = 0

        # Model (design matrix) counter:
        varCntMdl = 0

        # Voxel chunk counter:
        varCntChnk = 0

        while True:

            # Input dictionary to feed:
            dicIn = {objPlcHld01: aryDsgn[varCntMdl, :, :],
                     objPlcHld02: aryFunc[:, varCntChnk, :]}

            # Push to the queue:
            objSess.run(objEnQ, feed_dict=dicIn)

            varCntIt += 1

            # When the counter has reached the number of models
            # (design matrices), the next chunk of voxel is accessed
            # (and the loop through models starts again).
            varCntMdl += 1
            if varCntMdl == varNumMdl:
                varCntMdl = 0
                varCntChnk += 1

            # Stop if coordinator says stop:
            if objCoord.should_stop():
                break

            # Stop if all data has been put on the queue:
            elif varCntIt == (varNumTtl):
                break

    # ----------------------------------------------------------------
    # *** Preparations

    # print('-Tensorflow demo.')

    # Total number of graph calls:
    varNumTtl = varNumMdl * varNumChnk

    # Data to perform computations on.

    # Design matrices:
    aryDsgn = np.random.randn(varNumMdl,
                              varNumVol,
                              varNumBeta
                              ).astype(np.float32)

    # 'Functional data':
    aryFunc = np.random.randn(varNumVol,
                              varNumChnk,
                              varNumVoxPerChnk).astype(np.float32)

    # ----------------------------------------------------------------
    # *** Define the queue & the session

    print('---Defining graph')

    # Queue capacity:
    varCapQ = 10

    # The queue:
    objQ = tf.FIFOQueue(capacity=varCapQ,
                        dtypes=[tf.float32, tf.float32])

    # Method for getting queue size:
    objSzeQ = objQ.size()

    # Placeholder that are the input for the queue:
    objPlcHld01 = tf.placeholder(tf.float32,
                                 shape=[varNumVol, varNumBeta])
    objPlcHld02 = tf.placeholder(tf.float32,
                                 shape=[varNumVol, varNumVoxPerChnk])

    # The enqueue operation that puts data on the graph.
    objEnQ = objQ.enqueue([objPlcHld01, objPlcHld02])

    # Number of threads that will be created:
    varNumThrd = 1

    # The queue runner (places the enqueue operation on the queue?).
    objRunQ = tf.train.QueueRunner(objQ, [objEnQ] * varNumThrd)
    tf.train.add_queue_runner(objRunQ)

    # The tensor objects that are retrieved from the queue. These
    # function like placeholders for the data in the queue when
    # defining the graph.
    objIn01, objIn02 = objQ.dequeue()

    # Regularisation factor:
    #  varL2reg = 0.0

    # The computational graph.
    objGrph = tf.reduce_sum(
                            tf.abs(
                                   tf.subtract(
                                               tf.matmul(
                                                         objIn01,
                                                         tf.matmul(
                                                                   tf.matmul(
                                                                             tf.matrix_inverse(
                                                                                               tf.matmul(
                                                                                                         objIn01,
                                                                                                         objIn01,
                                                                                                         transpose_a=True,
                                                                                                         transpose_b=False
                                                                                                         )
                                                                                               ),
                                                                             objIn01,
                                                                             transpose_a=False,
                                                                             transpose_b=True
                                                                             ),
                                                                   objIn02
                                                                   )
                                                         ),
                                               objIn02),
                                   ),
                            axis=0,
                            )
#    objGrph = tf.matmul(
#                        tf.matmul(
#                                  tf.matrix_inverse(
#                                                    tf.matmul(
#                                                              objIn01,
#                                                              objIn01,
#                                                              transpose_a=True,
#                                                              transpose_b=False
#                                                              )
#                                                    ),
#                                  objIn01,
#                                  transpose_a=False,
#                                  transpose_b=True
#                                  ),
#                        objIn02
#                        )

    # Define session:
    objSess = tf.Session()

    # Coordinator needs to be initialised as well:
    objCoord = tf.train.Coordinator()

    # ----------------------------------------------------------------
    # *** Fill queue

    print('---Fill queue')

    # Buffer size (number of samples to put on queue before starting
    # execution of graph):
    varBuff = 10

    # Define & run extra thread with graph that places data on queue:
    objThrd = threading.Thread(target=funcPlcIn)
    objThrd.setDaemon(True)
    objThrd.start()

    # Stay in this while loop until the specified number of samples
    # (varBuffer) have been placed on the queue).
    varTmpSzeQ = 0
    while varTmpSzeQ < varBuff:
        varTmpSzeQ = objSess.run(objSzeQ)

    # ----------------------------------------------------------------
    # *** Run the graph

    print('---Run graph')

    # Variables need to be initialised:
    objSess.run(tf.global_variables_initializer())

    # Get time:
    varTme01 = time.time()

    # List for results:
    # lstRes = [None] * (varNumMdl * varNumChnk)

    # Initialise index (if we cannot use a for loop because range
    # object would be too large, we use a while loop instead).
    # idxIt = 0

    # Loop through input iterations:
    # while idxIt < varNumTtl:
    for idxIt in range(varNumTtl):

        # varTme04 = time.time()

        # Run main computational graph:
        vecTmp = objSess.run(objGrph)
        # lstRes[idxIt] = objSess.run(objGrph)

        # print(('---------Time for graph call: '
        #        + str(time.time() - varTme04)))

        # On every xth call, check number of elements on queue:
#        if (idxIt % 1000) == 0:
#
#            # Number of elements on queue:
#            varTmpSzeQ = objSess.run(objSzeQ)
#
#            strTmpMsg = ('------Iteration: '
#                         + str(idxIt)
#                         + ', number of elements on queue: '
#                         + str(varTmpSzeQ))
#
#            print(strTmpMsg)

        # idxIt += 1

    print(type(vecTmp))
    print(type(vecTmp[0]))
    print(vecTmp.shape)
    print(vecTmp[0:5, 0:5])

    # Stop threads.
    objCoord.request_stop()
    # objCoord.join(objThrds)
    objSess.close()

    # Get time:
    varTme02 = time.time()
    # varTme03 = np.around((varTme02 - varTme01), decimals=3)
    varTme03 = varTme02 - varTme01

    print(('---Time for running graph: '
           + str(np.around(varTme03, decimals=3))))

    return varTme03
    # ----------------------------------------------------------------


# Is this module is called directly?
if __name__ == "__main__":

    print('-GPU GLM fitting demo.')

    # Totel number of voxels to use:
    varNumVoxTtl = 2000

    # List with chunk sizes:
    lstNumChnk = [1]  # list(range(2, 31))

    # Resulting number of voxels per chunk:
    lstNumVoxPerChnk = [varNumVoxTtl / x for x in lstNumChnk]

    # Number of volumes:
    varNumVol = 2000

    # Number of predictors in the design matrix:
    varNumBeta = 5

    # Number of design matrices to loop through:
    varNumMdl = 1500

    # Number of scenarios:
    varNumScn = len(lstNumChnk)

    # Vector for timing results:
    vecTme = np.zeros((varNumScn))

    # Loop through scenarios:
    for idxScn in range(varNumScn):

        print(('--Scenario: ' + str(idxScn)))

        print(('--Number of voxels: '
               + str(lstNumChnk[idxScn] * lstNumVoxPerChnk[idxScn])))

        print(('--Number of voxels per chunk: '
               + str(lstNumVoxPerChnk[idxScn])))

        # Call to main function performing GLM fitting on GPU
        vecTme[idxScn] = funcGlmGpu(varNumVol=varNumVol,
                                    varNumChnk=lstNumChnk[idxScn],
                                    varNumVoxPerChnk=lstNumVoxPerChnk[idxScn],
                                    varNumBeta=varNumBeta,
                                    varNumMdl=varNumMdl)

    # Save results for inspection:
    #np.save('/home/john/Desktop/tmp/vecTme.npy',
    #        vecTme)
    #aryNumVoxPerChnk = np.array(lstNumVoxPerChnk)
    #np.save('/home/john/Desktop/tmp/aryNumVoxPerChnk.npy',
    #        aryNumVoxPerChnk)

    # ----------------------------------------------------------------
    # *** Create plot

    #plt.plot(aryNumVoxPerChnk, vecTme)
