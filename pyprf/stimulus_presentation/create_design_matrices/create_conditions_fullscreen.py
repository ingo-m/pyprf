# -*- coding: utf-8 -*-
"""Create design matrices for pRF mapping experiment."""



import os
import argparse
import numpy as np
from psychopy import gui, core


# import errno


def crt_design(dicParam):
    """
    Create design matrices for pRF mapping experiment.
    
    Parameters
    ----------
    dicParam : dictionary
        Dictionary containing parameters for creation of design matrix.
        
    Returns
    -------
    This function has no return values. A design matrix is created and saved
    in an npz file in the respective folder
    (~/pyprf/pyprf/stimulus_presentation/design_matrices/).

    """








    # number of TRs of rest during the experiment (if not desired set to zero)
    NrNullFixBetween = 0
    if NrNullFixBetween > 0:
        # avoid that two rest periods can occur immdetialey after each other?
        lgcRep = True
        # how many TRs should one rest preiod be? (default 1)
        NrOfTrPerNull = 3
    
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
            NullPos = np.random.choice(np.arange(1, len(conditions)),
                                       NrNullFixBetween, replace=False)
            NullPos = np.sort(NullPos)
            lgcRep = np.greater(np.sum(np.diff(NullPos) == 1), 0)
    
        # create null trials in between to be inserted into conditions
        NullTrialsInBetween = np.zeros(NrOfTrPerNull)[:, None]
        # insert null trials in between
        for i, idx in enumerate(NullPos):
            # adjustment to consider prev. iterations
            idx = idx + (i*NrOfTrPerNull)
            conditions = np.insert(conditions, idx, NullTrialsInBetween, axis=0)
    
    # add fixation blocks in beginning and end
    conditions = np.vstack((np.zeros((NrNullFixStart, 2)),
                            conditions,
                            np.zeros((NrNullFixEnd, 2))))
    
    # ----------------------------------------------------------------------------
    # Makeshift solution for nonsquare visual field: present more steps (NrOfSteps
    # is 20 instead of 12), but remove extra steps for horizontal bars (positions
    # 1 and 5).
    
    # We assume that the aspect ratio of the screen is 1920.0 / 1200.0. Horizontal
    # bars should only be presented at the central 62.5% percent of positions
    # relative to the extent of positions of vertical bars (because
    # 1200.0 / 1920.0 = 0.625).
    
    # Number of positions:
    vecSteps = np.arange(0, NrOfSteps)
    
    # Margin to leave out for low/high y-positions:
    varMarg = np.ceil((float(NrOfSteps) - (0.625 * float(NrOfSteps))) * 0.5)
    
    
    # New condition list (which will replace old list):
    lstCon = []
    for idx01 in range(conditions.shape[0]):
        # Position of current trial (i.e. current row):
        varTmpPos = conditions[idx01, 0]
        # Orientation of current trial (i.e. current row):
        varTmpOri = conditions[idx01, 1]
        # Check whether current trial has horizontal orientation:
        if ((varTmpOri == 1.0) or (varTmpOri == 5.0)):
            # Check whether horizontal orientation is presented outside of the
            # screen area:
            if ((varTmpPos < varMarg)
                    or ((float(NrOfSteps) - varMarg) <= varTmpPos)):
                # print((str(varTmpPos) + '   ' + str(varTmpOri)))
                pass
            else:
                # Horizontal orientation is within screen area, keep it:
                lstCon.append((varTmpPos, varTmpOri))
        else:
            # Orientation is not horizontal, keep it"
            lstCon.append((varTmpPos, varTmpOri))
    
    # Replace original conditions array with new array (without horizontal
    # condition outside of screen area):
    conditions = np.array(lstCon)
    # ----------------------------------------------------------------------------
    
    
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

if __name__ == "__main__":

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - open a GUI for user to specify design matrix
    # parameters?:
    objParser.add_argument('-gui',
                           # metavar='GUI',
                           choices=['True', 'False'],
                           default='True',
                           help='Open a GUI to set parameters for design \
                                 matrix?'
                           )


    # Add argument to namespace - open a GUI for user to specify design matrix
    # parameters?:
    objParser.add_argument('-filename',
                           # metavar='filename',
                           default=None,
                           help='Optional. Output file name for design \
                                 matrix. The  output file name can also be \
                                 set in the GUI.'
                           )

    # Namespace object containing arguments and values:
    objNspc = objParser.parse_args()

    # Get arguments from argument parser:
    strGui = objNspc.gui
    strFleNme = objNspc.filename
   
    # Dictionary with experiment parameters. 
    # - 'Number of bar orientations' = number of steps between 0 and 360 deg.
    # 
    dicParam = {'Output file name': 'Run_04_Fullscreen',
                'TR [s]': 2.0,
                'Target duration [s]': 0.3,
                'Number of bar orientations': 8,      # NrOfOrientation
                'Number of positions': 20,            # NrOfSteps
                'Number of blocks': 2,                # NrOfBlocks
                'Initial rest period [volumes]': 10,  # NrNullFixStart
                'Final rest period [volumes]': 10}    # NrNullFixEnd

    if not(strFleNme is None):
        
        # If an input file name is provided, put it into the dictionary (as
        # default, can still be overwritten by user).
        dicParam['Output file name'] = strFleNme

    if strGui == 'True':

        # Pop-up GUI to let the user select parameters:
        objGui = gui.DlgFromDict(dictionary=dicParam,
                                 title='Design Matrix Parameters')
           
        # Close if user presses 'cancel':
        if objGui.OK is False:
            core.quit()

    # Output path ('~/pyprf/pyprf/stimulus_presentation/design_matrices/'):
    strPth = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    strPth = os.path.join(strPth, 'design_matrices')

    # Add output path to dictionary.
    dicParam['Output path'] = strPth

    print(dicParam)

    #crt_design(dicParam)

