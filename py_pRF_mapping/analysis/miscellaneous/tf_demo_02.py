# -*- coding: utf-8 -*-
"""
Simple tensorflow demo using queue to place input data on graph.

This version uses a separate graph, running in a separate thread, to
place data on the queue. High GPU utilisation can be achieved with
this configuration.
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


# --------------------------------------------------------------------
# *** Function definition

# Define queue-feeding-function that will run in extra thread:
def funcPlcIn():
    """Place data on queue."""
    # Iteration counter:
    idxCnt = 0

    while True:

        # Feed example to Tensorflow placeholder
        dicIn = {objPlcHld01: lstIn01[idxCnt],
                 objPlcHld02: lstIn02[idxCnt]}

        # Push to the queue:
        objSess.run(objEnQ, feed_dict=dicIn)

        idxCnt += 1

        # Stop if coordinator says stop:
        if objCoord.should_stop():
            break

        # Stop if all data has been put on the queue:
        elif idxCnt == varNumIt:
            break


# --------------------------------------------------------------------
# *** Preparations

print('-Tensorflow demo.')

# Data to perform computations on. First dimension is number of
# iterations.
varNumIt = 1000
aryIn = np.ones((varNumIt, 3000, 2000), dtype=np.float32)
vecIn = np.arange(1, (varNumIt + 1), dtype=np.float32)
vecIn = np.reshape(vecIn, (varNumIt, 1))

# Put input data into lists (needed as input for feed_dict for graph
# that feeds queue):
lstIn01 = [None] * varNumIt
lstIn02 = [None] * varNumIt
for idxIt in range(varNumIt):
    lstIn01[idxIt] = aryIn[idxIt, :, :]
    lstIn02[idxIt] = vecIn[idxIt]

# Remember array dimensions:
varDim01 = aryIn.shape[1]
varDim02 = aryIn.shape[2]
varDim03 = vecIn[0].shape[0]

del(aryIn)
del(vecIn)

# --------------------------------------------------------------------
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
                             shape=[varDim01, varDim02])
objPlcHld02 = tf.placeholder(tf.float32,
                             shape=[varDim03])

# The enqueue operation that puts data on the graph.
objEnQ = objQ.enqueue([objPlcHld01, objPlcHld02])

# Number of threads that will be created:
varNumThrd = 1

# The queue runner (places the enqueue operation on the queue?).
objRunQ = tf.train.QueueRunner(objQ, [objEnQ] * varNumThrd)
tf.train.add_queue_runner(objRunQ)

# The tensor objects that are retrieved from the queue. These function
# like placeholders for the data in the queue when defining the graph.
objIn01, objIn02 = objQ.dequeue()

# The computational graph. Just some intense nonsense computation.
objGrph = tf.reduce_sum(
                        tf.divide(
                                  tf.multiply(
                                              tf.abs(
                                                     tf.add(
                                                            tf.multiply(objIn01,
                                                                        objIn01),
                                                            objIn01
                                                            )
                                                     ),
                                              objIn02
                                              ),
                                  tf.multiply(
                                              tf.add(
                                                     tf.multiply(objIn01,
                                                                 objIn01),
                                                     objIn01
                                                     ),
                                              objIn02
                                              )
                                  )
                        )

# The following graph returns larger object, has lower GPU utilisation and is
# slower (although there is one fewer caluclation).
#objGrph = tf.divide(
#                    tf.multiply(
#                                tf.abs(
#                                       tf.add(
#                                              tf.multiply(objIn01,
#                                                          objIn01),
#                                              objIn01
#                                              )
#                                       ),
#                                objIn02
#                                ),
#                    tf.multiply(
#                                tf.add(
#                                       tf.multiply(objIn01,
#                                                   objIn01),
#                                       objIn01
#                                       ),
#                                objIn02
#                                )
#                    )


# Define session:
objSess = tf.Session()

# Coordinator needs to be initialised as well:
objCoord = tf.train.Coordinator()


# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# *** Run the graph

print('---Run graph')

# Variables need to be initialised:
objSess.run(tf.global_variables_initializer())

# Get time:
varTme01 = time.time()

# List for results:
lstRes = [None] * varNumIt

# Loop through input iterations:
for idxIt in range(varNumIt):

    # Run main computational graph and put results in list:
    # varTme04 = time.time()

    # Run main computational graph and put results in list:
    lstRes[idxIt] = objSess.run(objGrph)
    # lstRes[0] = objSess.run(objGrph)
    # objSess.run(objGrph)

    # print(('---------Time for graph call: '
    #        + str(time.time() - varTme04)))

    # On every xth call, check number of elements on queue:
    if (idxIt % 50) == 0:

        # Number of elements on queue:
        varTmpSzeQ = objSess.run(objSzeQ)

        strTmpMsg = ('------Iteration: '
                     + str(idxIt)
                     + ', number of elements on queue: '
                     + str(varTmpSzeQ))

        print(strTmpMsg)

#print(type(lstRes))
#print(len(lstRes))
#
#print(type(lstRes[0]))
#print(lstRes)

# Stop threads.
objCoord.request_stop()
#objCoord.join(objThrds)
objSess.close()

# Get time:
varTme02 = time.time()
varTme03 = np.around((varTme02 - varTme01), decimals=3)

print(('---Time for running graph: ' + str(varTme03)))
# --------------------------------------------------------------------
