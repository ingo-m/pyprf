# -*- coding: utf-8 -*-
"""
Simple tensorflow demo using queue to place input data on graph.

The problem with this way of placing data on the graph is that the
enqueue_many operation loads all data onto GPU memory, which does not work for
large datasets.
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
import tensorflow as tf
import numpy as np

print('-Tensorflow demo.')

# The computation that will be performed on the graph is a multiplication of
# arrays with scalars. The array to be multiplied is 2D. In order to pass
# data to the graph with enqueue_many, they are sliced along the first
# dimension. We want to iterate through many multiplications, so both the 2D
# array and the scaller get the number of iterations as the 0th dimension.
varNumIt = 20
aryIn = np.ones((varNumIt, 1024, 1024), dtype=np.float32)
vecIn = np.arange(0, varNumIt, dtype=np.float32)
vecIn = np.reshape(vecIn, (varNumIt, 1))

print('---Defining graph')

# Define the queue & the graph
with tf.variable_scope('queue'):

    # The queue:
    objQ = tf.FIFOQueue(capacity=5, dtypes=[tf.float32, tf.float32])

    # The enqueue operation that puts data on the graph. The array and the
    # vector are sliced along the 0th dimension automatically.
    objEnQ = objQ.enqueue_many([aryIn, vecIn])

    # Number of threads that will be created:
    varNumThrd = 1

    # The queue runner (places the enqueue operation on the queue?).
    objRunQ = tf.train.QueueRunner(objQ, [objEnQ] * varNumThrd)
    tf.train.add_queue_runner(objRunQ)

    # The tensor objects that are retrieved from the queue. These function like
    # placeholders for the data in the queue when defining the graph.
    objIn01, objIn02 = objQ.dequeue()

    # The computational graph (multiplies the sliced array with the sliced
    # vector, retrieves the first element and prints it).
    objGrph = tf.Print(objIn02, [tf.multiply(objIn01, objIn02)[0][0]])

print('---Run graph')

# Get time:
varTme01 = time.time()

# Define session:
with tf.Session() as objSess:

    # Variables need to be initialised:
    objSess.run(tf.global_variables_initializer())

    # Coordinator needs to be initialised as well:
    objCoord = tf.train.Coordinator()
    objThrds = tf.train.start_queue_runners(coord=objCoord)

    # Run the graph:
    for idxIt in range(varNumIt):
        obj01 = objSess.run(objGrph)

    # Stop threads.
    objCoord.request_stop()
    objCoord.join(objThrds)

# Get time:
varTme02 = time.time()
varTme03 = np.around((varTme02 - varTme01), decimals=3)

print(('---Time for running graph: ' + str(varTme03)))
