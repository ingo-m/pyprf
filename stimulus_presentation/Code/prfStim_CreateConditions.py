# -*- coding: utf-8 -*-
"""Create conditions."""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import pickle

# set parameters
TR = 2.94
TargetDuration = 0.3

NrOfBlocks = 4
NrOfOrientation = 8
NrOfSteps = 12

NrNullFixStart = 8
NrNullFixEnd = 8

NrOfVols = (NrOfBlocks * NrOfOrientation * NrOfSteps + NrNullFixStart
            + NrNullFixEnd)

# create arrays for position and orientation
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

# add fixation blocks in beginning and end
conditions = np.vstack((aryPos, aryOri)).T
conditions = np.vstack((np.zeros((NrNullFixStart, 2)),
                        conditions,
                        np.zeros((NrNullFixEnd, 2))))

# prepare targets
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


# prepare random target onset delay
BlockOnsetinSec = np.arange(len(conditions)) * TR
TargetOnsetinSec = BlockOnsetinSec[targets]
TargetOnsetDelayinSec = np.random.uniform(0.1,
                                          TR-TargetDuration,
                                          size=NrOfTargets)
TargetOnsetinSec = TargetOnsetinSec + TargetOnsetDelayinSec

# create dictionary for saving to pickle
array_run1 = {'Conditions': conditions,
              'TargetOnsetinSec': TargetOnsetinSec,
              'TR': TR,
              'TargetDuration': TargetDuration,
              'NrOfSteps': NrOfSteps,
              'NrOfVols': NrOfVols,
              }


# save dictionary to pickle
folderpath = '/media/sf_D_DRIVE/ParamContrast/pRFStimuli/prfStim/prfStim/Conditions'
filename1 = os.path.join(folderpath, 'Conditions_run01.pickle')

with open(filename1, 'wb') as handle:
    pickle.dump(array_run1, handle)
