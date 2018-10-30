#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Test of h5py for creation of hdf5 files."""

import numpy as np
import time
import h5py

# Get time:
varTme01 = time.time()

# Create hdf5 file:
fleDsgn = h5py.File('/home/john/Desktop/tmp/aryPrfTc.hdf5', 'w')

varNumMdls = 40 * 40 * 64
varNumCon = 2
varNumVol = 6356

# Create dataset within hdf5 file:
dtsPrfTc = fleDsgn.create_dataset('pRF_time_courses',
                                  (varNumMdls,
                                   varNumCon,
                                   varNumVol),
                                  dtype=np.float32)

for idxCon in range(varNumCon):
    for idxMdl in range(varNumMdls):

        # Place random data on dataset:
        dtsPrfTc[idxMdl, idxCon, :] = np.random.randn(varNumVol
                                                      ).astype(np.float32)

# Close file:
fleDsgn.close()

# Get time:
varTme02 = time.time()
varTme03 = np.around((varTme02 - varTme01), decimals=3)

print(('Time: ' + str(varTme03)))
