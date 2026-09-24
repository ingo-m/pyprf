# -*- coding: utf-8 -*-
"""Create design matrices for pRF mapping experiment."""

import os
import csv
import argparse
import numpy as np


def crt_design(dicParam, lgcOvwr=False):
    """
    Create design matrices for pRF mapping experiment.

    Parameters
    ----------
    dicParam : dictionary
        Dictionary containing parameters for creation of design matrix.
    lgcOvwr : bool
        Whether to overwrite an existing design matrix with the same name. If
        False (default), an error is raised if the design matrix exists.

    Returns
    -------
    strPthNpz : str
        Path of the design matrix. The design matrix is saved in an npz file
        (and in human-readable format in a txt file) in the respective folder
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

    # Number of orientations between for the bar stimulus:
    varNumOri = int(dicParam['Number of bar orientations'])

    # Number of position steps for the bar stimulus:
    varNumPosX = int(dicParam['Number of bar positions on x-axis'])
    varNumPosY = int(dicParam['Number of bar positions on y-axis'])

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

    # Stimulus contrasts:
    lstCon = dicParam['Stimulus contrasts']

    # Path of output file:
    strPthNpz = os.path.join(strPth, strFleNme) + '.npz'

    # Check whether file already exists:
    if os.path.isfile(strPthNpz) and not lgcOvwr:
        strMsg = ('Design matrix already exists: ' + strPthNpz
                  + ' Please delete it or choose a different file name.')
        raise FileExistsError(strMsg)

    # *************************************************************************
    # *** Preparations

    # Duration of rest period [volumes] - only relevant if varNumRest is not
    # zero.
    varDurRest = 3

    # Number of contrast levels:
    varNumCon = len(lstCon)

    # *************************************************************************
    # *** Stimulus orientation & position

    # Number of x-positions and y-positions should both be even or both be odd,
    # in order for bar positions to cover the scren regularly.
    if (varNumPosX % 2) == 0:

        # Even number of x-positions.

        if (varNumPosY % 2) == 1:

            # Odd number of y-positions.
            strTmp = ('Even number of x-positions, and odd number of '
                      + 'y-positions. Will add one more y-position.')
            print(strTmp)
            varNumPosY = varNumPosY + 1

    else:

        # Odd number of x-positions.

        if (varNumPosY % 2) == 0:

            # Evem number of y-positions.
            strTmp = ('Odd number of x-positions, and even number of '
                      + 'y-positions. Will add one more y-position.')
            print(strTmp)
            varNumPosY = varNumPosY + 1

    #  Create arrays for position, orientation, and contrast:
    aryOri = np.empty(0)
    aryPos = np.empty(0)
    aryCon = np.empty(0)

    # List of orientations. Psychopy orientation convention: "Orientation
    # convention is like a clock: 0 is vertical, and positive values rotate
    # clockwise." Actually, 0 is the positive x-axis.
    if varNumOri == 4:
        # Orientations are coded as follows: horizontal = 0.0, vertical = 90.0,
        # lower left to upper right = 45.0, upper left to lower right = 135.0.
        lstOri = [0.0, 45.0, 90.0, 135.0]
    elif varNumOri == 2:
        # If number of orientations is set to two, only vertical and horizontal
        # orientations are presented (i.e. no oblique bars).
        lstOri = [0.0, 90.0]

    # Loop through blocks (repititions of orientation & position):
    for idxBlk in range(varNumBlk):

        # Array for orientations in current block:
        aryOriBlock = np.array(lstOri)

        # Randomise order of orientations in current block:
        np.random.shuffle(aryOriBlock)

        # Each orientation is repeated as many times as there are positions (it
        # is assumed that the x-dimension - i.e. screen width - is as least as
        # large as the y-dimension).
        aryOriBlock = np.repeat(aryOriBlock, (varNumPosX * varNumCon))

        # Array for positions within current block:
        # aryPosBlock = np.empty(0)

        # Array for positions & contrasts within current block:
        aryPosConBlock = np.empty(0).reshape((0, 2))

        # Loop through orientations:
        for idxOri in range(len(lstOri)):

            # Array for positions:
            aryPosTemp = np.arange(varNumPosX)

            # Repeat positions as many time as there are contrast levels:
            aryPosTemp = np.tile(aryPosTemp, varNumCon)

            # Array for contrast levels:
            aryConTmp = np.repeat(lstCon, varNumPosX)

            # Stack positions and contrasts levels, in order to randomise them
            # together:
            aryPosConTmp = np.vstack((aryPosTemp, aryConTmp)).T

            # Pseudorandomisation: Don't present the same position twice
            # in direct succession.

            # Switch for pseudorandomisation:
            lgcTmp = True

            while lgcTmp:

                # Randomise order of positions (shuffles the array along the
                # first axis).
                np.random.shuffle(aryPosConTmp)

                # Minimum difference between successive position codes (if this
                # value is zero, this means that the same position occurs
                # twice in a row).
                varDiffMin = np.min(np.abs(np.diff(aryPosConTmp[:, 0])))

                # Break loop if the same position does not occur twice in a
                # row.
                if 0.0 < varDiffMin:
                    lgcTmp = False

            # aryPosBlock = np.append(aryPosBlock, aryPosTemp)
            aryPosConBlock = np.append(aryPosConBlock, aryPosConTmp, axis=0)

        # Put randomised orientations/positions/contrasts into run arrays:
        aryOri = np.append(aryOri, aryOriBlock)
        aryPos = np.append(aryPos, aryPosConBlock[:, 0])
        aryCon = np.append(aryCon, aryPosConBlock[:, 1])

    # Number of volumes so far (will be updated when rest blocks are inserted).
    varNumVol = aryCon.shape[0]

    # Array for complete design matrix. The design matrix consists of four
    # columns, containing the following information: (1) Stimulus or rest? (2)
    # Bar position (3) Bar orientation (4) Bar contrast (or possibly another
    # stimulus feature implemented in the stimulation script).
    aryDsg = np.zeros((varNumVol, 4))

    # Put orientations and positions into design matrix:
    aryDsg[:, 1] = aryPos
    aryDsg[:, 2] = aryOri
    aryDsg[:, 3] = aryCon

    # At this point, the design matrix contains only stimulus events (i.e. no
    # rest blocks). The first column of the design matrix is used to code for
    # stimulus/rest (redundant but useful in stimulation script). Here, we set
    # the entire first column of the design matrix to 'one', signifying a
    # stimulus event. Rest blocks that will be added will be coded as 'zero'.
    aryDsg[:, 0] = 1.0

    # *************************************************************************
    # *** Full screen mode

    if lgcFull:

        # Remove horizontal bar positions that are not on screen.

        # Margin to leave out for low/high y-positions:
        varMarg = float(varNumPosX - varNumPosY) * 0.5

        # New condition list (which will replace old list):
        lstCon = []

        # Loop through volumes:
        for idxVol in range(varNumVol):

            # Stimulus/rest code of current trial (i.e. current row):
            varTmpStim = aryDsg[idxVol, 0]

            # Position of current trial (i.e. current row):
            varTmpPos = aryDsg[idxVol, 1]

            # Orientation of current trial (i.e. current row):
            varTmpOri = aryDsg[idxVol, 2]

            # Contrast of current trial (i.e. current row):
            varTmpCon = aryDsg[idxVol, 3]

            # Check whether current trial has horizontal orientation:
            # if ((varTmpOri == 0.0) or (varTmpOri == 180.0)):
            if (varTmpOri == 0.0):

                # Check whether horizontal orientation is presented outside of
                # the screen area. Of the positions 0 to (varNumPosX - 1), the
                # central varNumPosY positions are kept (i.e. positions varMarg
                # to (varNumPosX - varMarg - 1)).
                if ((varTmpPos < varMarg)
                        or ((float(varNumPosX) - varMarg) <= varTmpPos)):

                    # print((str(varTmpPos) + '   ' + str(varTmpOri)))
                    pass

                else:

                    # Horizontal orientation is within screen area, keep it:
                    lstCon.append((varTmpStim,
                                   varTmpPos,
                                   varTmpOri,
                                   varTmpCon))

            else:

                # Orientation is not horizontal, keep it:
                lstCon.append((varTmpStim, varTmpPos, varTmpOri, varTmpCon))

        # Replace original aryDsg array with new array (without horizontal
        # condition outside of screen area):
        aryDsg = np.array(lstCon)

    # Update number of volumes:
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

            else:

                # If there is only one rest block, there is no need to check
                # minimum distance.
                lgcRep = False

        # Zeros for rest blocks to be inserted into design matrix:
        aryZeros = np.zeros((varDurRest, 4))

        # Insert rest blocks into design matrix:
        for idxRest, varRstStrt in enumerate(vecNullIdx):

            # Adjustment insertion index to consider previous iterations:
            varRstStrtTmp = varRstStrt + (idxRest * varDurRest)

            # Insert into design matrix:
            aryDsg = np.insert(aryDsg, varRstStrtTmp, aryZeros, axis=0)

    # Add fixation blocks at beginning and end of run:
    aryDsg = np.vstack((np.zeros((varDurRestStrt, 4)),
                        aryDsg,
                        np.zeros((varDurRestEnd, 4))))

    # Update number of volumes:
    varNumVol = aryDsg.shape[0]

    # *************************************************************************
    # *** Randomise target events

    # Number of volumes on which target events could occur (not during initial
    # and final rest periods):
    varNumVolTrgt = (varNumVol - varDurRestStrt - varDurRestEnd)

    # Number of target events to present (on average, one every varIti
    # seconds):
    varNumTrgt = int(np.around(((float(varNumVolTrgt) * varTr) / varIti)))

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

    # Save design matrix to npz file:
    np.savez(strPthNpz,
             aryDsg=aryDsg,
             vecTrgt=vecTrgt,
             lgcFull=lgcFull,
             varTr=varTr,
             varNumVol=varNumVol,
             varNumOri=varNumOri,
             varNumPosX=varNumPosX,
             varNumPosY=varNumPosY,
             varNumTrgt=varNumTrgt,
             varIti=varIti)

    # Save information in human-readable format
    lstCsv = []
    lstCsv.append('* * *')
    lstCsv.append('Design matrix')
    lstCsv.append('    Columns:')
    lstCsv.append('    (1) Stimulus or rest?')
    lstCsv.append('    (2) Bar position')
    lstCsv.append('    (3) Bar orientation')
    lstCsv.append('    (4) Bar contrast')
    for strTmp in aryDsg.tolist():
        lstCsv.append(strTmp)
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
    lstCsv.append('Number of bar positions on x-axis')
    lstCsv.append(str(varNumPosX))
    lstCsv.append('* * *')
    lstCsv.append('Number of bar positions on y-axis')
    lstCsv.append(str(varNumPosY))
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

    return strPthNpz


def func_free_name(strPth):
    """
    Get a name for a new design matrix.

    Returns the first name of the form 'Run_01', 'Run_02', ... that is not used
    by a design matrix in the directory `strPth`.
    """
    idxRun = 1
    while os.path.isfile(os.path.join(strPth,
                                      'Run_' + str(idxRun).zfill(2) + '.npz')):
        idxRun += 1
    return 'Run_' + str(idxRun).zfill(2)


# *****************************************************************************

if __name__ == "__main__":

    # PsychoPy is only needed for the GUI. (Importing it here allows to use
    # `crt_design` without PsychoPy, e.g. for testing.)
    from psychopy import gui, core

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - open a GUI for user to specify design matrix
    # parameters?:
    objParser.add_argument('-gui',
                           # metavar='GUI',
                           choices=['True', 'False'],
                           default='True',
                           help='Open a GUI to set parameters for design \
                                 matrix? If not, default parameters are \
                                 used.'
                           )

    # Add argument to namespace - output file name:
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
    lgcGui = (objNspc.gui == 'True')
    strFleNme = objNspc.filename

    # Output path ('~/pyprf/pyprf/stimulus_presentation/design_matrices/'):
    strPth = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    strPth = os.path.join(strPth, 'design_matrices')

    # Default output file name (if no file name is provided): The first name
    # of the form 'Run_01', 'Run_02', ... that is not used by an existing
    # design matrix.
    if strFleNme is None:
        strFleNme = func_free_name(strPth)

    # Dictionary with experiment parameters (in the order in which they are
    # shown in the GUI).
    dicParam = {'Output file name': strFleNme,
                'TR [s]': 2.079,
                'Number of bar orientations': [4, 2],
                'Number of bar positions on x-axis': 14,
                'Number of bar positions on y-axis': 8,
                'Number of blocks': 4,
                'Number of rest trials': 1,
                'Inter-trial interval for targets [s]': 15.0,
                'Initial rest period [volumes]': 10,
                'Final rest period [volumes]': 10,
                'Full screen:': [True, False],
                'Stimulus contrasts': [[1.0], [0.05, 1.0]]}

    # Luminance constrasts for LGN pRF mapping experiment:
    # Maximum contrast = 0.99, at pixel value 1.0
    # Low contrast = 0.1  (precisely: 0.0975), at pixel value: 0.05

    if lgcGui:

        # Pop-up GUI to let the user select parameters:
        objGui = gui.DlgFromDict(dictionary=dicParam,
                                 title='Design Matrix Parameters',
                                 order=list(dicParam.keys()))

        # Close if user presses 'cancel':
        if not objGui.OK:
            core.quit()

    else:

        # Without GUI, use the first option of parameters with several
        # options:
        for strKey, varVal in dicParam.items():
            if isinstance(varVal, list):
                dicParam[strKey] = varVal[0]

    # Add output path to dictionary.
    dicParam['Output path'] = strPth

    # If no file name is given, use default name:
    dicParam['Output file name'] = (str(dicParam['Output file name']).strip()
                                    or func_free_name(strPth))

    # If a design matrix with this name already exists, the user can choose a
    # different name, or overwrite the existing design matrix.
    lgcOvwr = False
    while (os.path.isfile(os.path.join(strPth, dicParam['Output file name'])
                          + '.npz')
           and not lgcOvwr):

        strFleNme = dicParam['Output file name']
        strMsg = ('There already is a design matrix called "' + strFleNme
                  + '" in the folder:\n' + strPth)

        if not lgcGui:
            print(strMsg + '\nPlease choose a different file name.')
            core.quit()

        objDlg = gui.Dlg(title='Design matrix already exists')
        objDlg.addText(strMsg + '\n\nPlease choose a different name, or '
                       + 'overwrite the existing design matrix.')
        objDlg.addField('Output file name', func_free_name(strPth))
        objDlg.addField('Overwrite "' + strFleNme + '" instead', False)
        # Hide note on required fields (PsychoPy Qt GUI; there are none):
        if hasattr(objDlg, 'validate'):
            objDlg.validate()
        lstData = objDlg.show()

        # Close if user presses 'cancel':
        if not objDlg.OK:
            core.quit()

        if lstData[1]:
            lgcOvwr = True
        else:
            dicParam['Output file name'] = (str(lstData[0]).strip()
                                            or func_free_name(strPth))

    # Create design matrix:
    strPthNpz = crt_design(dicParam, lgcOvwr=lgcOvwr)

    # Show summary. The number of volumes and the TR are needed to set up the
    # fMRI sequence.
    objNpz = np.load(strPthNpz)
    varTr = float(objNpz['varTr'])
    varNumVol = int(objNpz['varNumVol'])
    varDur = int(np.around(varTr * varNumVol))
    strMsg = ('Design matrix saved as:\n' + strPthNpz + '\n\n'
              + 'Volume TR: ' + str(varTr) + ' s\n'
              + 'Number of volumes: ' + str(varNumVol) + '\n'
              + 'Duration: ' + str(varDur // 60) + ' min '
              + str(varDur % 60) + ' s\n\n'
              + 'The fMRI sequence has to acquire (at least) '
              + str(varNumVol) + ' volumes, with a TR of ' + str(varTr)
              + ' s.')
    print(strMsg)
    if lgcGui:
        objDlg = gui.Dlg(title='Design matrix created')
        objDlg.addText(strMsg)
        if hasattr(objDlg, 'validate'):
            objDlg.validate()
        objDlg.show()
