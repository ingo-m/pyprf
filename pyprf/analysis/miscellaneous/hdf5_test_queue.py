#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Test of h5py for creation of hdf5 files."""

import numpy as np
import time
import h5py
import threading
import queue


# Get time:
varTme01 = time.time()


# -------------------------------------------------------------------------
# *** Queue-feeding-function that will run in extra thread
def funcPlcIn(varNumCon, varNumMdls, objQ):
    """Place data on queue."""
    # Loop through models:
    for idxCon in range(varNumCon):
        for idxMdl in range(varNumMdls):

            # Take data from queue, and put in hdf5 file:
            dtsPrfTc[idxMdl, idxCon, :] = objQ.get()


# -----------------------------------------------------------------
# *** Prepare queue

varNumMdls = 40 * 40 * 64
varNumCon = 2
varNumVol = 6356

# Buffer size:
varBuff = 10

# Create FIFO queue:
objQ = queue.Queue(maxsize=varBuff)

# Define & run extra thread with graph that places data on queue:
objThrd = threading.Thread(target=funcPlcIn,
                           args=(varNumCon, varNumMdls, objQ))
objThrd.setDaemon(True)
objThrd.start()

# -----------------------------------------------------------------
# *** Fill hdf5 file

# Create hdf5 file:
fleDsgn = h5py.File('/home/john/Desktop/tmp/aryPrfTc.hdf5', 'w')

# Create dataset within hdf5 file:
dtsPrfTc = fleDsgn.create_dataset('pRF_time_courses',
                                  (varNumMdls,
                                   varNumCon,
                                   varNumVol),
                                  dtype=np.float32)

# Loop through models:
for idxCon in range(varNumCon):

    print(('Condition ' + str(idxCon)))

    for idxMdl in range(varNumMdls):

        # Place random data on queue:
        objQ.put(np.random.randn(varNumVol).astype(np.float32))

print('Finished random number creation')

# Join queue:
# objQ.join()

print('Joined queue')

# Close thread:
objThrd.join()

# Close file:
fleDsgn.close()

# Get time:
varTme02 = time.time()
varTme03 = np.around((varTme02 - varTme01), decimals=3)

print(('Time: ' + str(varTme03)))
