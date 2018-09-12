# -*- coding: utf-8 -*-
"""Create design matrices for pRF mapping experiment."""

import os
import csv
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
    # *************************************************************************
    # *** Get parameters from dictionary

    # File name of design matrix:
    strFleNme = dicParam['Output file name']

    # Output path for design matrix:
    strPth = dicParam['Output path']

    # Volume TR [s]:
    varTr = float(dicParam['TR [s]'])

    # Target duration [s]:
    # varTrgtDur = float(dicParam['Target duration [s]'])

    # Inter-trial interval between target events [s]:
    varIti = float(dicParam['Inter-trial interval for targets [s]'])

    # Number of orientations between 0 and 360 degree for the bar stimulus:
    varNumOri = int(dicParam['Number of bar orientations'])

    # Number of position steps for the bar stimulus:
    varNumPos = int(dicParam['Number of positions'])

    # Number of stimulus blocks (each positions and orientation occurs once
    # per stimulus block):
    varNumBlk = int(dicParam['Number of blocks'])

    # Number of rest trials:
    varNumRest = int(dicParam['Number of rest trials'])

    # Duration of initial rest period [volumes]:
    varDurRestStrt = int(dicParam['Initial rest period [volumes]'])

    # Duration of final rest period [volumes]:
    varDurRestEnd = int(dicParam['Final rest period [volumes]'])

    # Full screen stimuli?
    lgcFull = dicParam['Full screen:']

    # *************************************************************************
    # *** Preparations

    # Duration of rest period [volumes] - only relevant if varNumRest is not
    # zero.
    varDurRest = 3

    # *************************************************************************
    # *** Stimulus orientation & position

    #  Create arrays for position and orientation
    aryOri = np.empty(0)
    aryPos = np.empty(0)

    # Loop through blocks (repititions of orientation & position):
    for idxBlk in range(varNumBlk):

        # Array for orientations in current block:
        aryOriBlock = np.arange(1, (varNumOri + 1))

        # Randomise order of orientations in current block:
        np.random.shuffle(aryOriBlock)

        # Each orientation is repeated as many times as there are positions:
        aryOriBlock = np.repeat(aryOriBlock, varNumPos)

        # Array for positions within current block:
        aryPosBlock = np.empty(0)

        # Loop through orientations:
        for idxOri in range(varNumOri):

            # Array for positions:
            aryPosTemp = np.arange(varNumPos)

            # Randomise order of positions:
            np.random.shuffle(aryPosTemp)

            aryPosBlock = np.append(aryPosBlock, aryPosTemp)

        # Put randomised orientation and positions into run arrays:
        aryOri = np.append(aryOri, aryOriBlock)
        aryPos = np.append(aryPos, aryPosBlock)

    # Array for complete design matrix
    aryDsg = np.vstack((aryPos, aryOri)).T

    # Number of volumes:
    varNumVol = aryDsg.shape[0]

    # *************************************************************************
    # *** Add rest blocks

    # Add rest blocks?
    if varNumRest > 0:

        # Avoid that two rest periods can occur immdetialey after each other.
        lgcRep = True

        # First and last volumes on which an additional rest block can occur.
        # (We would not want the additional rest block to occur at the very
        # beginning or end (where there are already initial and final rest
        # blocks).
        varLmtRst01 = varDurRestStrt + 10
        varLmtRst02 = varNumVol - varDurRestEnd - 10

        while lgcRep:

            # Vector of volume indices at which the random rest blocks could
            # occur.
            vecNullIdx = np.random.choice(
                                          np.arange(varLmtRst01,
                                                    varLmtRst02,
                                                    1,
                                                    dtype=np.int16),
                                          varNumRest,
                                          replace=False
                                          )

            # Only check for minimum distance between rest blocks if there are
            # at least two rest blocks:
            if varNumRest > 1:

                # Sort selected indices at which additional rest blocks will be
                # shown:
                vecNullIdx = np.sort(vecNullIdx)

                # Check whether the time difference between two consecutive
                # rest blocks is at least 10 volumes:
                lgcDiff = np.all(np.greater_equal(np.diff(vecNullIdx), 10))

                # Only continue if the time difference between rest blocks is
                # too small"
                lgcRep = not(lgcDiff)

        # Zeros for rest blocks to be inserted into design matrix:
        aryZeros = np.zeros(varDurRest)[:, None]

        # Insert rest blocks into design matrix:
        for idxRest, varRstStrt in enumerate(vecNullIdx):

            # Adjustment insertion index to consider previous iterations:
            varRstStrtTmp = varRstStrt + (idxRest * varDurRest)

            # Insert into design matrix:
            aryDsg = np.insert(aryDsg, varRstStrtTmp, aryZeros, axis=0)

    # Add fixation blocks at beginning and end of run:
    aryDsg = np.vstack((np.zeros((varDurRestStrt, 2)),
                        aryDsg,
                        np.zeros((varDurRestEnd, 2))))

    # Update number of volumes:
    varNumVol = aryDsg.shape[0]

    # *************************************************************************
    # *** Full screen mode

    if lgcFull:

        # Makeshift solution for nonsquare visual field: present more steps
        # (e.g. 20 instead of 12), but remove extra steps for horizontal bars
        # (positions 1 and 5).

        # We assume that the aspect ratio of the screen is 1920.0 / 1200.0.
        # Horizontal bars should only be presented at the central 62.5% percent
        # of positions relative to the extent of positions of vertical bars
        # (because 1200.0 / 1920.0 = 0.625).

        # Number of positions along vertical axis:
        varNumPosX = int(np.ceil(float(varNumPos) * (1920.0 / 1200.0)))

        # Number of positions:
        # vecSteps = np.arange(0, varNumPosX)

        # Margin to leave out for low/high y-positions:
        varMarg = np.ceil(
                          (float(varNumPosX) - (0.625 * float(varNumPosX)))
                          * 0.5
                          )

        # New condition list (which will replace old list):
        lstCon = []

        # Loop through volumes:
        for idxVol in range(varNumVol):

            # Position of current trial (i.e. current row):
            varTmpPos = aryDsg[idxVol, 0]

            # Orientation of current trial (i.e. current row):
            varTmpOri = aryDsg[idxVol, 1]

            # Check whether current trial has horizontal orientation:
            if ((varTmpOri == 1.0) or (varTmpOri == 5.0)):

                # Check whether horizontal orientation is presented outside of
                # the screen area:
                if ((varTmpPos < varMarg)
                        or ((float(varNumPos) - varMarg) <= varTmpPos)):

                    # print((str(varTmpPos) + '   ' + str(varTmpOri)))
                    pass

                else:

                    # Horizontal orientation is within screen area, keep it:
                    lstCon.append((varTmpPos, varTmpOri))

            else:

                # Orientation is not horizontal, keep it:
                lstCon.append((varTmpPos, varTmpOri))

        # Replace original aryDsg array with new array (without horizontal
        # condition outside of screen area):
        aryDsg = np.array(lstCon)

    # *************************************************************************
    # *** Randomise target events

    # Number of target events to present (on average, one every varIti
    # seconds):
    varNumTrgt = int(np.around(((float(varNumVol) * varTr) / varIti)))

    # Earliest & latest allowed time for target event in seconds:
    varLmtTrgt01 = float(varDurRestStrt) * varTr + 2.0
    varLmtTrgt02 = float(varNumVol - varDurRestEnd) * varTr - 2.0

    lgcRep = True

    while lgcRep:

        # Vector of volume indices at which the random target events could
        # occur.
        vecTrgt = np.random.choice(
                                   np.arange(varLmtTrgt01,
                                             varLmtTrgt02,
                                             0.1,
                                             dtype=np.float32),
                                   varNumTrgt,
                                   replace=False
                                   )

        # Sort selected indices at which additional rest blocks will be
        # shown:
        vecTrgt = np.sort(vecTrgt)

        # Check whether the time difference between two consecutive
        # target events is at least 3 seconds:
        lgcDiff01 = np.all(np.greater_equal(np.diff(vecTrgt), 3.0))

        # Check whether the time difference between two consecutive
        # target events is not more than 35 seconds:
        lgcDiff02 = np.all(np.less_equal(np.diff(vecTrgt), 35.0))

        # Only continue if the time difference between rest blocks is
        # too small"
        lgcRep = not(np.multiply(lgcDiff01, lgcDiff02))

    # *************************************************************************
    # *** Save design matrix

    # Concatenate output path and file name:
    strPthNpz = os.path.join(strPth, strFleNme) + '.npz'

    # Check whether file already exists:
    if os.path.isfile(strPthNpz):

        strMsg = ('WARNING: File already exists. Please delete or choose '
                  + 'different file name.')

        print(strMsg)

    else:

        # Save design matrix to npz file:
        np.savez(strPthNpz,
                 aryDsg=aryDsg,
                 vecTrgt=vecTrgt,
                 lgcFull=lgcFull,
                 varTr=varTr,
                 varNumVol=varNumVol,
                 varNumOri=varNumOri,
                 varNumPos=varNumPos,
                 varNumTrgt=varNumTrgt,
                 varIti=varIti)

        # Save information in human-readable format
        lstCsv = []
        lstCsv.append('* * *')
        lstCsv.append('Design matrix')
        lstCsv.append(aryDsg.tolist())
        lstCsv.append('* * *')
        lstCsv.append('Target events')
        lstCsv.append(list(vecTrgt))
        lstCsv.append('* * *')
        lstCsv.append('Full screen mode')
        lstCsv.append(str(lgcFull))
        lstCsv.append('* * *')
        lstCsv.append('Volume TR [s]')
        lstCsv.append(str(varTr))
        lstCsv.append('* * *')
        lstCsv.append('Number of volumes')
        lstCsv.append(str(varNumVol))
        lstCsv.append('* * *')
        lstCsv.append('Number of bar orientations')
        lstCsv.append(str(varNumOri))
        lstCsv.append('* * *')
        lstCsv.append('Number of bar positions')
        lstCsv.append(str(varNumPos))
        lstCsv.append('* * *')
        lstCsv.append('Number of target events')
        lstCsv.append(str(varNumTrgt))
        lstCsv.append('* * *')
        lstCsv.append('Inter-trial interval for target events [s]')
        lstCsv.append(str(varIti))

        # Output path:
        strPthTxt = os.path.join(strPth, strFleNme) + '.txt'

        # Create output csv object:
        objCsv = open(strPthTxt, 'w')

        # Save list to disk:
        csvOt = csv.writer(objCsv, lineterminator='\n')

        # Write output list data to file (row by row):
        for strTmp in lstCsv:
            csvOt.writerow([strTmp])

        # Close:
        objCsv.close()


# *****************************************************************************

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
    dicParam = {'Output file name': 'Run_01',
                'TR [s]': 2.0,
                # 'Target duration [s]': 0.3,
                'Number of bar orientations': 8,
                'Number of positions': 12,
                'Number of blocks': 2,
                'Number of rest trials': 0,
                'Inter-trial interval for targets [s]': 15.0,
                'Initial rest period [volumes]': 10,
                'Final rest period [volumes]': 10,
                'Full screen:': [True, False]}

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

    crt_design(dicParam)
