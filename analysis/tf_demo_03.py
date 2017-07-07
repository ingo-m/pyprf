# -*- coding: utf-8 -*-
"""
Simple tensorflow demo using queue to place input data on graph.

This version uses a separate graph, running in a separate thread, to place data
on the queue. GLM fitting is tested, looping through voxels and models
(inefficient).
"""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it
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

import time
import threading
import tensorflow as tf
import numpy as np


# -----------------------------------------------------------------------------
# *** Function definition

# Define queue-feeding-function that will run in extra thread:
def funcPlcIn():
    """Function for placing data on queue."""

    # Iteration counter:
    idxCnt = 0

    # Model counter:
    varCntMdl = 0

    # Voxel counter:
    varCntVox = 0

    while True:

        aryTmp01 = np.copy(ary01[varCntMdl, :, :])
        aryTmp02 = np.copy(ary02[:, varCntVox])
        aryTmp02 = np.reshape(aryTmp02, (aryTmp02.shape[0], 1))

        # Feed example to Tensorflow placeholder
#        dicIn = {objPlcHld01: ary01[varCntMdl, :, :],
#                 objPlcHld02: ary02[:, varCntVox]}
        dicIn = {objPlcHld01: aryTmp01,
                 objPlcHld02: aryTmp02}

        # Push to the queue:
        objSess.run(objEnQ, feed_dict=dicIn)

        idxCnt += 1

        varCntMdl +=1
        if varCntMdl == varNumMdl:
            varCntMdl = 0
            varCntVox += 1

        # Stop if coordinator says stop:
        if objCoord.should_stop():
            break

        # Stop if all data has been put on the queue:        
        elif idxCnt == (varNumMdl * varNumVox):
            break


# -----------------------------------------------------------------------------
# *** Preparations

print('-Tensorflow demo.')

varNumVol = 400
varNumVox = 50000
varNumBeta = 2
varNumMdl = 10000
strPath = '/home/john/Desktop/tmp/ary0{}.npy'
strSwitch = 'create'

# Data to perform computations on. First dimension is number of iterations.
if strSwitch == 'create':

    # 'Model time courses':
    ary01 = np.random.randn(varNumMdl, varNumVol, varNumBeta).astype(np.float32)

    # 'Functional data':
    ary02 = np.random.randn(varNumVol, varNumVox).astype(np.float32)

    np.save(strPath.format('1'), ary01)
    np.save(strPath.format('2'), ary02)

elif strSwitch == 'load':

    ary01 = np.load(strPath.format('1')).astype(np.float32)
    ary02 = np.load(strPath.format('2')).astype(np.float32)


# -----------------------------------------------------------------------------
# *** Define the queue & the session

print('---Defining graph')

# Queue capacity:
varCapQ = 10

# The queue:
objQ = tf.FIFOQueue(capacity=varCapQ, dtypes=[tf.float32, tf.float32])

# Method for getting queue size:
objSzeQ = objQ.size()

# Placeholder that are the input for the queue:
objPlcHld01 = tf.placeholder(tf.float32,
                             shape=[varNumVol, varNumBeta])
objPlcHld02 = tf.placeholder(tf.float32,
                             shape=[varNumVol, 1])

# The enqueue operation that puts data on the graph.
objEnQ = objQ.enqueue([objPlcHld01, objPlcHld02])

# Number of threads that will be created:
varNumThrd = 1

# The queue runner (places the enqueue operation on the queue?).
objRunQ = tf.train.QueueRunner(objQ, [objEnQ] * varNumThrd)
tf.train.add_queue_runner(objRunQ)

# The tensor objects that are retrieved from the queue. These function like
# placeholders for the data in the queue when defining the graph.
objIn01, objIn02 = objQ.dequeue()

# Regularisation factor:
varL2reg = 0.0

# The computational graph. Just some intense nonsense computation.
objGrph = tf.reduce_sum(
                        tf.abs(
                               tf.subtract(
                                           tf.matmul(
                                                     objIn01,
                                                     tf.matrix_solve_ls( \
                                                                        objIn01, objIn02,
                                                                        varL2reg,
                                                                        fast=True
                                                                        )
                                                     ),
                                           objIn02),
                               ),
                        axis=0
                        )




# Define session:
objSess = tf.Session()

# Coordinator needs to be initialised as well:
objCoord = tf.train.Coordinator()


# -----------------------------------------------------------------------------
# *** Fill queue

print('---Fill queue')

# Buffer size (number of samples to put on queue before starting execution of
# graph):
varBuff = 10

# Define & run extra thread with graph that places data on queue:
objThrd = threading.Thread(target=funcPlcIn)
objThrd.setDaemon(True)
objThrd.start()

# Stay in this while loop until the specified number of samples (varBuffer)
# have been placed on the queue).
varTmpSzeQ = 0
while varTmpSzeQ < varBuff:
    varTmpSzeQ = objSess.run(objSzeQ)


# -----------------------------------------------------------------------------
# *** Run the graph

print('---Run graph')

# Variables need to be initialised:
objSess.run(tf.global_variables_initializer())

# Get time:
varTme01 = time.time()

# List for results:
# lstRes = [None] * (varNumMdl * varNumVox)

# Loop through input iterations:
for idxIt in range(varNumMdl * varNumVox):

    # Run main computational graph and put results in list:
    # varTme04 = time.time()

    # Run main computational graph and put results in list:
    vecTmp = objSess.run(objGrph)
    # lstRes[0] = objSess.run(objGrph)
    # objSess.run(objGrph)

    # print(('---------Time for graph call: '
    #        + str(time.time() - varTme04)))

    # On every xth call, check number of elements on queue:
    if (idxIt % 100) == 0:

        # Number of elements on queue:
        varTmpSzeQ = objSess.run(objSzeQ)

        strTmpMsg = ('------Iteration: '
                     + str(idxIt)
                     + ', number of elements on queue: '
                     + str(varTmpSzeQ))

        print(strTmpMsg)

print(type(vecTmp))
print(type(vecTmp[0]))
print(vecTmp[0].shape)

# Stop threads.
objCoord.request_stop()
#objCoord.join(objThrds)
objSess.close()

# Get time:
varTme02 = time.time()
varTme03 = np.around((varTme02 - varTme01), decimals=3)

print(('---Time for running graph: ' + str(varTme03)))
# -----------------------------------------------------------------------------
