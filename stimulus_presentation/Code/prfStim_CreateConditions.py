# -*- coding: utf-8 -*-
"""Create conditions."""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import errno
import pickle

# %% Set parameters

# set folder for stimulation condition
OutFolderPath = '/home/faruk/Documents/test/Conditions'
OutFileName = 'Conditions_run01.pickle'

# set the TR
TR = 1.20
# set the target duration in [s]
TargetDuration = 0.3

# set the number of bar orientations (i.e. # of steps between 0 and 360 deg)
NrOfOrientation = 8
# set nr of steps when traversing the visual field, e.g. from left to right
NrOfSteps = 12
# nr of blocks per run (one block contains all possible steps x orientations)
NrOfBlocks = 4

# number of TRs of rest at start of experiment
NrNullFixStart = 8
# number of TRs of rest at end of experiment
NrNullFixEnd = 8
# number of TRs of rest during the experiment (if not desired set to zero)
NrNullFixBetween = 50
if NrNullFixBetween > 0:
    # avoid that two rest periods can occur immdetialey after each other?
    lgcRep = True
    # how many TRs should one rest preiod be? (default 1)
    NrOfTrPerNull = 3

NrOfVolsExclNull = (NrOfBlocks * NrOfOrientation * NrOfSteps)

# %% Create arrays for position and orientation
aryOri = np.empty(0)
aryPos = np.empty(0)
for indBlock in np.arange(NrOfBlocks):
    aryOriBlock = np.arange(1, NrOfOrientation+1)
    np.random.shuffle(aryOriBlock)
    aryOriBlock = np.repeat(aryOriBlock, NrOfSteps)
    aryPosBlock = np.empty(0)
    for ind in np.arange(NrOfOrientation):
        aryPosTemp = np.arange(NrOfSteps)
        np.random.shuffle(aryPosTemp)
        aryPosBlock = np.append(aryPosBlock, aryPosTemp)
    aryOri = np.append(aryOri, aryOriBlock)
    aryPos = np.append(aryPos, aryPosBlock)

# add fixation blocks inbetween the experiment
conditions = np.vstack((aryPos, aryOri)).T

if NrNullFixBetween > 0:
    while lgcRep:
        NullPos = np.random.choice(np.arange(1, NrOfVolsExclNull),
                                   NrNullFixBetween, replace=False)
        lgcRep = np.greater(np.sum(np.diff(np.sort(NullPos)) == 1), 0)

# insert null trials in between
conditions = np.insert(conditions, NullPos, np.array([0, 0]),
                       axis=0)

# add fixation blocks in beginning and end
conditions = np.vstack((np.zeros((NrNullFixStart, 2)),
                        conditions,
                        np.zeros((NrNullFixEnd, 2))))

# %% Prepare target arrays
NrOfTargets = int(len(conditions)/10)
targets = np.zeros(len(conditions))
lgcRep = True
while lgcRep:
    targetPos = np.random.choice(np.arange(NrNullFixStart,
                                 len(conditions)-NrNullFixEnd), NrOfTargets,
                                 replace=False)
    lgcRep = np.greater(np.sum(np.diff(np.sort(targetPos)) == 1), 0)
targets[targetPos] = 1
assert NrOfTargets == np.sum(targets)
targets = targets.astype(bool)

# %% Prepare random target onset delay
BlockOnsetinSec = np.arange(len(conditions)) * TR
TargetOnsetinSec = BlockOnsetinSec[targets]
TargetOnsetDelayinSec = np.random.uniform(0.1,
                                          TR-TargetDuration,
                                          size=NrOfTargets)
TargetOnsetinSec = TargetOnsetinSec + TargetOnsetDelayinSec

# %% Create dictionary for saving to pickle
array_run1 = {'Conditions': conditions,
              'TargetOnsetinSec': TargetOnsetinSec,
              'TR': TR,
              'TargetDuration': TargetDuration,
              'NrOfSteps': NrOfSteps,
              'NrOfVols': len(conditions),
              }

# %% Save dictionary to pickle
OutFile = os.path.join(OutFolderPath, OutFileName)

try:
    os.makedirs(OutFolderPath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
    with open(OutFile, 'wb') as handle:
        pickle.dump(array_run1, handle)
