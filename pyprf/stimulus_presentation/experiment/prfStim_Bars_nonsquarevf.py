# -*- coding: utf-8 -*-
"""
Stimulus presentation for pRF mapping.

Present retinotopic mapping stimuli with Psychopy.

This version: Non-square visual field coverage, i.e. bar stimuli all over the
              screen.
"""

# Part of pyprf library
# Copyright (C) 2016  Marian Schneider & Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
import pickle
import datetime
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from psychopy.misc import pol2cart
from itertools import cycle
from PIL import Image


def prf_stim():
    """
    Present stimuli for population receptive field mapping.

    If in logging mode, this script
    creates a stimulus log of the stimuli used for the pRF mapping that can be
    used for the pRF finding analysis of the pyprf library. The stimuli are saved
    as png files, where each png represents the status of visual stimulation for
    one TR (the png files contain modified screenshots of the visual stimulus,
    and can be directly be loaded into the py_pRF_mapping pipepline.
    """

    # *****************************************************************************
    # *** Logging




    # Logging mode:
    if dicExpInfo['Logging mode'] == 'Yes':
        lgcLogMde = True
    else:
        lgcLogMde = False
    # *****************************************************************************


    # *****************************************************************************
    # *** Logging

    # Set clock:
    objClck = core.Clock()

    # Control the logging of participant responses:
    varSwtRspLog = 0

    # The key that the participant has to press after a target event:
    strTrgtKey = '1'

    # Counter for correct/incorrect responses:
    varCntHit = 0  # Counter for hits
    varCntMis = 0  # Counter for misses

    # Set clock for logging:
    logging.setDefaultClock(objClck)

    # Add time stamp and experiment name to metadata:
    dicExpInfo['Date'] = data.getDateStr().encode('utf-8')
    dicExpInfo['Experiment_Name'] = strExpNme

    # Path of this file:
    strPthMain = os.path.dirname(os.path.abspath(__file__))

    # Get parent path:
    strPthPrnt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Path of logging folder (parent to subject folder):
    strPthLog = (strPthPrnt
                 + os.path.sep
                 + 'log')

    # If it does not exist, create subject folder for logging information
    # pertaining to this session:
    if not os.path.isdir(strPthLog):
        os.makedirs(strPthLog)

    # Path of subject folder:
    strPthSub = (strPthLog
                 + os.path.sep
                 + str(dicExpInfo['Subject_ID'])
                 )

    # If it does not exist, create subject folder for logging information
    # pertaining to this session:
    if not os.path.isdir(strPthSub):
        os.makedirs(strPthSub)

    # Name of log file:
    strPthLog = (strPthSub
                 + os.path.sep
                 + '{}_{}_Run_{}_{}'.format(dicExpInfo['Subject_ID'],
                                            dicExpInfo['Experiment_Name'],
                                            dicExpInfo['Run'],
                                            dicExpInfo['Date'])
                 )

    # Create a log file and set logging verbosity:
    fleLog = logging.LogFile(strPthLog + '.log', level=logging.DATA)

    # Log parent path:
    fleLog.write('Parent path: ' + strPthPrnt + '\n')

    # Log condition:
    fleLog.write('Subject_ID: ' + dicExpInfo['Subject_ID'] + '\n')
    fleLog.write('Run: ' + dicExpInfo['Run'] + '\n')
    fleLog.write('Test mode: ' + dicExpInfo['Test mode'] + '\n')

    # Set console logging verbosity:
    logging.console.setLevel(logging.WARNING)

    # Array for logging of key presses:
    aryKeys = np.array([], dtype=np.float32)
    # *****************************************************************************

    # Load stimulus parameters from npz file.
    objNpz = np.load(strPthNpz)

    # Get design matrix (for bar positions and orientation):
    aryDsg = objNpz['aryDsg']

    # Vector with times of target events:
    vecTrgt = objNpz['vecTrgt']

    # Full screen mode? If no, bar stimuli are restricted to a central square.
    # If yes, bars appear on the entire screen. This parameter is set when
    # creating the design matrix.
    lgcFull = objNpz['lgcFull']

    # Number of volumes:
    varNumVol = objNpz['varNumVol']

    # NUmber of bar positions:
    varNumPos = objNpz['varNumPos']

    # Number of target events:
    varNumTrgt = objNpz['varNumTrgt']

    # Average inter-trial interval for target events:
    varIti = objNpz['varIti']














    # *****************************************************************************
    # *** Setup

    # Create monitor object:
    objMon = monitors.Monitor('Screen_7T_NOVA_32_Channel_Coil',
                              width=varMonWdth,
                              distance=varMonDist)

    # Convert size in pixels to size in degrees (given the monitor settings):
    varDegCover = misc.pix2deg(varPix, objMon)

    # Set size of monitor:
    objMon.setSizePix([varPixX, varPixY])

    # Log monitor info:
    fleLog.write(('Monitor distance: varMonDist = '
                  + str(varMonDist)
                  + ' cm'
                  + '\n'))
    fleLog.write(('Monitor width: varMonWdth = '
                  + str(varMonWdth)
                  + ' cm'
                  + '\n'))
    fleLog.write(('Monitor width: varPixX = '
                  + str(varPixX)
                  + ' pixels'
                  + '\n'))
    fleLog.write(('Monitor height: varPixY = '
                  + str(varPixY)
                  + ' pixels'
                  + '\n'))

    # Set screen:
    objWin = visual.Window(
        size=(varPixX, varPixY),
        screen=0,
        winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
        allowGUI=False,
        allowStencil=True,
        fullscr=True,
        monitor=objMon,
        color=lstBckgrd,
        colorSpace='rgb',
        units='deg',
        blendMode='avg'
        )
    # *****************************************************************************


    # *****************************************************************************





    # %%
    """aryDsg"""
    # retrieve aryDsg from pickle file (stored in folder aryDsg)
    str_path_aryDsg = str_path_parent_up + os.path.sep + 'aryDsg' + \
        os.path.sep + 'aryDsg_run' + str(dicExp['run']) + '.pickle'
    with open(str_path_aryDsg, 'rb') as handle:
        arrays = pickle.load(handle)

    aryDsg = arrays["aryDsg"]
    aryDsg = aryDsg.astype(int)
    vecTrgt = arrays["vecTrgt"]
    TargetDur = arrays["TargetDuration"]

    # If in logging mode, only present stimuli very briefly:
    if lgcLogMde:
        # Note: If the 'varTr' is set too low in logging mode, frames are
        # dropped and the stimuli do not get logged properly.
        varTr = 0.2
    # Otherwise, use actual volume TR:
    else:
        # Volume TR:
        varTr = objNpz['varTr']

    varNumPos = arrays["varNumPos"]
    varNumVol = arrays["varNumVol"]
    print('TARGETS: ')
    print vecTrgt

    # calculate
    Offset = varPix/varNumPos/2
    aryOri = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = np.linspace(-varPix/2+Offset, varPix/2-Offset, varNumPos)

    aryPosPix = []
    for ori in [90, 45, 180, 135, 270, 225, 0, 315]:
        temp = zip(*pol2cart(np.tile(ori, varNumPos), distances))
        aryPosPix.append(temp)

    # create array to log key pressed events
    TriggerPressedArray = np.array([])
    TargetPressedArray = np.array([])

    logFile.write('aryDsg=' + unicode(aryDsg) + '\n')
    logFile.write('vecTrgt=' + unicode(vecTrgt) + '\n')
    logFile.write('TargetDur=' + unicode(TargetDur) + '\n')

    # %%
    """STIMULI"""

    # INITIALISE SOME STIMULI
    grating = visual.GratingStim(
        objWin,
        tex="sqrXsqr",
        color=[1.0, 1.0, 1.0],
        colorSpace='rgb',
        opacity=1.0,
        size=(2*varPixX, varPix/varNumPos),
        sf=(varSptlFrq/(varPix/varNumPos), varSptlFrq/(varPix/varNumPos)),
        ori=0,
        autoLog=False,
        interpolate=False,
        )

    # fixation dot
    dotFix = visual.Circle(
        objWin,
        autoLog=False,
        name='dotFix',
        radius=2,
        fillColor=[1.0, 0.0, 0.0],
        lineColor=[1.0, 0.0, 0.0],)

    dotFixSurround = visual.Circle(
        objWin,
        autoLog=False,
        name='dotFixSurround',
        radius=7,
        fillColor=[0.5, 0.5, 0.0],
        lineColor=[0.0, 0.0, 0.0],)

    # fixation grid
    Circle = visual.Polygon(
        win=objWin,
        autoLog=False,
        name='Circle',
        edges=90,
        ori=0,
        units='deg',
        pos=[0, 0],
        lineWidth=2,
        lineColor=[1.0, 1.0, 1.0],
        lineColorSpace='rgb',
        fillColor=None,
        fillColorSpace='rgb',
        opacity=1,
        interpolate=True)
    Line = visual.Line(
        win=objWin,
        autoLog=False,
        name='Line',
        start=(-varPixY, 0),
        end=(varPixY, 0),
        pos=[0, 0],
        lineWidth=2,
        lineColor=[1.0, 1.0, 1.0],
        lineColorSpace='rgb',
        fillColor=None,
        fillColorSpace='rgb',
        opacity=1,
        interpolate=True,)
    # initialisation method
    message = visual.TextStim(
        objWin,
        autoLog=False,
        text='Condition',
        height=30,
        pos=(400, 400)
        )
    triggerText = visual.TextStim(
        win=objWin,
        autoLog=False,
        color='white',
        height=30,
        text='Experiment will start soon. \n Waiting for scanner',)
    targetText = visual.TextStim(
        win=objWin,
        autoLog=False,
        color='white',
        height=30)

    vertices = [(varPix/2, varPix/2), (-varPix/2, varPix/2),
                (-varPix/2, -varPix/2), (varPix/2, -varPix/2)]
    aperture = visual.Aperture(objWin,
                               autoLog=False,
                               shape=vertices)  # try shape='square'
    aperture.enabled = False

    # %%
    """TIME AND TIMING PARAMETERS"""

    # get screen refresh rate
    refr_rate = objWin.getActualFrameRate()  # get screen refresh rate
    if refr_rate is not None:
        frameDur = 1.0/round(refr_rate)
    else:
        frameDur = 1.0/60.0  # couldn't get a reliable measure so guess
    logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
    logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

    # set durations
    durations = np.arange(varTr, varTr*varNumVol + varTr, varTr)
    totalTime = varTr*varNumVol

    # how many frames b or w? derive from reversal frequency
    numFrame = int(round((1/(varTmpFrq*2))/frameDur))

    # create clock and Landolt clock
    clock = core.Clock()
    logging.setDefaultClock(clock)

    # %%
    """FUNCTIONS"""
    # update flicker in a square wave fashion
    # with every frame
    SquareArray = np.hstack((
        np.tile(1, numFrame),
        np.tile(-1, numFrame)
        ))
    SquareCycle = cycle(SquareArray)


    def squFlicker():
        """What does this function do? Why is it defined here?."""
        mContrast = SquareCycle.next()
        return mContrast


    # %% Logging mode preparations
    if lgcLogMde:

        print('Logging mode')

        # Calculate area to crop in y-dimension, at top and bottom (should be zero
        # is # full extend of screeen height is used):
        varCrpY = int(np.around((float(varPixY) - float(varPix)) * 0.5))

        print(('Stimulus log will be cropped by '
               + str(varCrpY)
               + ' in y-direction (screen height).'
               ))

        # Calculate area to crop in x-dimension, at left and right (would be zero
        # is full extend of screeen width was used):
        varCrpX = int(np.around((float(varPixX) - float(varPix)) * 0.5))

        print(('Stimulus log will be cropped by '
               + str(varCrpX)
               + ' in x-direction (screen width).'
               ))

        # Temporary array for screenshots, at full screen size and containing RGB
        # values (needed to obtain buffer content from psychopy):
        aryBuff = np.zeros((varPixY, varPixX, 3), dtype=np.int16)

        # Prepare array for screenshots. One value per pixel per volume; since the
        # stimuli are greyscale we discard 2nd and 3rd RGB dimension. Also, there
        # is no need to represent the entire screen, just the part of the screen
        # that is actually stimulated (this is typically a square at the centre of
        # the screen, flanked by unstimulated areas on the left and right side).
        aryFrames = np.zeros((varPixY, varPixX, varNumVol), dtype=np.int16)

        # Make sure that varPix is of interger type:
        varPix = int(np.around(varPix))

        # Counter for screenshots:
        idxFrame = 0

    # %%
    """RENDER_LOOP"""
    # Create Counters
    i = 0
    # give the system time to settle
    core.wait(1)

    if not(lgcLogMde):
        # wait for scanner trigger
        triggerText.draw()
        objWin.flip()
        event.waitKeys(keyList=['5'], timeStamped=False)

    # reset clocks
    clock.reset()
    logging.data('StartOfRun' + unicode(dicExp['run']))

    while clock.getTime() < totalTime:  # noqa

        # get key for motion direction
        keyPos = aryDsg[i, 0]
        keyOri = aryDsg[i, 1]

        # get direction
        if 0 < keyOri < 9:
            grating.setOpacity(1)
            grating.setOri(aryOri[keyOri-1])
            grating.setPos(aryPosPix[keyOri-1][keyPos])
        else:  # static
            grating.setOpacity(0)

        while clock.getTime() < durations[i]:
            # aperture.enabled = True
            # draw fixation grid (circles and lines)
            if not lgcLogMde:
                Circle.setSize((varDegCover*0.2, varDegCover*0.2))
                Circle.draw()
                Circle.setSize((varDegCover*0.4, varDegCover*0.4))
                Circle.draw()
                Circle.setSize((varDegCover*0.6, varDegCover*0.6))
                Circle.draw()
                Circle.setSize((varDegCover*0.8, varDegCover*0.8))
                Circle.draw()
                # subtract 0.1 here so that ring is not exactly at outer border
                Circle.setSize((varDegCover-0.1, varDegCover-0.1))
                Circle.draw()
                Line.setOri(0)
                Line.draw()
                Line.setOri(45)
                Line.draw()
                Line.setOri(90)
                Line.draw()
                Line.setOri(135)
                Line.draw()

            # update contrast flicker with square wave
            y = squFlicker()
            grating.contrast = y

            grating.draw()
            aperture.enabled = False

            # decide whether to draw target
            # first time in target interval? reset target counter to 0!
            if (
                (sum(clock.getTime() >= vecTrgt)
                 + sum(clock.getTime() < vecTrgt + 0.3)
                 ) == len(vecTrgt) + 1):
                # display target!
                # change color fix dot surround to red
                dotFixSurround.fillColor = [0.5, 0.0, 0.0]
                dotFixSurround.lineColor = [0.5, 0.0, 0.0]
            # dont display target!
            else:
                # keep color fix dot surround yellow
                dotFixSurround.fillColor = [0.5, 0.5, 0.0]
                dotFixSurround.lineColor = [0.5, 0.5, 0.0]

            if not lgcLogMde:
                # draw fixation point surround
                dotFixSurround.draw()
                # draw fixation point
                dotFix.draw()

            # draw frame
            objWin.flip()

            # handle key presses each frame
            for key in event.getKeys():
                if key in ['escape', 'q']:
                    logging.data(msg='User pressed quit')
                    objWin.close()
                    core.quit()
                elif key[0] in ['5']:
                    logging.data(msg='Scanner trigger')
                    TriggerPressedArray = np.append(TriggerPressedArray,
                                                    clock.getTime())
                elif key in ['1']:
                    logging.data(msg='Key1 pressed')
                    TargetPressedArray = np.append(TargetPressedArray,
                                                   clock.getTime())

        i = i + 1

        # %% Save screenshots to array
        if lgcLogMde:

            print(('---Frame '
                  + str(idxFrame)
                  + ' out of '
                  + str(int(varNumVol))))

            # Temporary array for single frame (3 values per pixel - RGB):
            aryBuff[:, :, :] = objWin.getMovieFrame(buffer='front')

            # print(type(aryBuff))
            # print('type(aryBuff)')
            # print('aryBuff.shape')
            # print(aryBuff.shape)

            # We only save one value per pixel per volume (because the stimuli are
            # greyscale we discard 2nd and 3rd RGB dimension):
            aryRgb = aryBuff[:, :, 0]

            # We only save the central square area that contains the stimulus:
            aryFrames[:, :, idxFrame] = np.copy(aryRgb)
            # np.copy(aryRgb[varCrpY:(varCrpY + varPix),
            #                varCrpX:(varCrpX + varPix)])
            idxFrame = idxFrame + 1

    logging.data('EndOfRun' + unicode(dicExp['run']) + '\n')

    # %%
    """TARGET DETECTION RESULTS"""

    # calculate target detection results
    # create an array 'targetDetected' for showing which targets were detected
    targetDetected = np.zeros(len(vecTrgt))
    if len(TargetPressedArray) == 0:
        # if no buttons were pressed
        print "No keys were pressed/registered"
        targetsDet = 0
    else:
        # if buttons were pressed:
        for index, target in enumerate(vecTrgt):
            for TimeKeyPress in TargetPressedArray:
                if (float(TimeKeyPress) >= float(target) and
                        float(TimeKeyPress) <= float(target) + 2):
                    targetDetected[index] = 1

    logging.data('ArrayOfDetectedTargets' + unicode(targetDetected))
    print 'Array Of Detected Targets: ' + str(targetDetected)

    # number of detected targets
    targetsDet = sum(targetDetected)
    logging.data('NumberOfDetectedTargets' + unicode(targetsDet))
    # detection ratio
    DetectRatio = targetsDet/len(targetDetected)
    logging.data('RatioOfDetectedTargets' + unicode(DetectRatio))

    # display target detection results to participant
    resultText = ('You have detected '
                  + str(int(np.around(targetsDet, decimals=0)))
                  + ' out of '
                  + str(len(vecTrgt))
                  + ' targets.')

    print resultText
    logging.data(resultText)
    # also display a motivational slogan
    if DetectRatio >= 0.95:
        feedbackText = 'Excellent! Keep up the good work'
    elif DetectRatio < 0.95 and DetectRatio > 0.85:
        feedbackText = 'Well done! Keep up the good work'
    elif DetectRatio < 0.8 and DetectRatio > 0.65:
        feedbackText = 'Please try to focus more'
    else:
        feedbackText = 'You really need to focus more!'

    targetText.setText(resultText+'\n'+feedbackText)
    logFile.write(unicode(resultText) + '\n')
    logFile.write(unicode(feedbackText) + '\n')
    targetText.draw()
    objWin.flip()
    core.wait(5)

    # %%
    """CLOSE DISPLAY"""
    objWin.close()

    # %%
    """SAVE DATA"""

    # create python dictionary
    output = {'ExperimentName': dicExp['strExpNme'],
              'Date': dicExp['date'],
              'SubjectID': dicExp['participant'],
              'Run_Number': dicExp['run'],
              'aryDsg': aryDsg,
              'TriggerPresses': TriggerPressedArray,
              'TargetPresses': TargetPressedArray,
              }
    try:
        # save dictionary as a pickle in output folder
        misc.toFile(outFileName + '.pickle', output)
        print 'Output Data saved as: ' + outFileName + '.pickle'
    except:
        print '(OUTPUT folder could not be created.)'

    # Save screenshots (logging mode)
    if lgcLogMde:

        print('Saving screenshots')

        # Target directory for frames (screenshots):
        strPthFrm = (dataFolderName + os.path.sep + 'Frames')

        # Check whether directory for frames exists, if not create it:
        lgcDir = os.path.isdir(strPthFrm)
        # If directory does not exist, create it:
        if not(lgcDir):
            # Create direcotry for segments:
            os.mkdir(strPthFrm)

        # Save stimulus frame array to npy file:
        aryFrames = aryFrames.astype(np.int16)
        np.savez_compressed((strPthFrm
                             + os.path.sep
                             + 'stimFramesRun'
                             + dicExp['run']),
                            aryFrames=aryFrames)

        # Maximum intensity of output PNG:
        varScle = 255

        # Rescale array:
        aryFrames = np.logical_or(np.equal(aryFrames, np.min(aryFrames)),
                                  np.equal(aryFrames, np.max(aryFrames))
                                  )
        aryFrames = np.multiply(aryFrames, varScle)
        aryFrames = aryFrames.astype(np.uint8)

        # Loop through volumes and save PNGs:
        for idxVol in range(varNumVol):

            print(('---Frame '
                  + str(idxVol)
                  + ' out of '
                  + str(int(varNumVol))))

            # Create image: TODO: compatibility with new Pillow version?
            im = Image.fromarray(aryFrames[:, :, idxVol])

            # File name (with leading zeros, e.g. '*_004' or '*_042'). For
            # consistency with earlier versions, the numbering of frames (PNG
            # files  corresponding to fMRI volumes) starts at '1' (not at '0').
            strTmpPth = (strPthFrm
                         + os.path.sep
                         + 'run_'
                         + dicExp['run']
                         + '_frame_'
                         + str(idxVol + 1).zfill(3)
                         + '.png')

            # Save image to disk:
            im.save(strTmpPth)

            # aryRgb = np.swapaxes(aryRgb, 0, 1)
            # aryRgb = np.swapaxes(aryRgb, 1, 2)

    # %%
    """FINISH"""
    core.quit()

# *****************************************************************************

if __name__ == "__main__":

    # *****************************************************************************
    # *** Settings



    # Frequency of stimulus bar in Hz:
    varTmpFrq = 4.0

    # Sptial frequency of stimulus (cycles per degree):
    varSptlFrq = 1.5

    # Size of area covered by bars (in pixels):
    varPix = 1920

    # Distance between observer and monitor [cm]:
    varMonDist = 99.0  # [99.0] for 7T scanner

    # Width of monitor [cm]:
    varMonWdth = 30.0  # [30.0] for 7T scanner

    # Width of monitor [pixels]:
    varPixX = 1920  # [1920.0] for 7T scanner

    # Height of monitor [pixels]:
    varPixY = 1200  # [1200.0] for 7T scanner

    # Background colour:
    lstBckgrd = [-0.7, -0.7, -0.7]
    # *****************************************************************************

# Name of experiment:
strExpNme = 'pRF_mapping'

# Get date string as default session name:
strDate = str(datetime.datetime.now())
lstDate = strDate[0:10].split('-')
strDate = (lstDate[0] + lstDate[1] + lstDate[2])

# Dictionary with experiment parameters:
dicExpInfo = {'Subject_ID': strDate,
              'Run': '04_nonsquarevf',
              'Logging mode': ['No', 'Yes']}

# Pop-up GUI to let the user select parameters:
objGui = gui.DlgFromDict(dictionary=dicExpInfo,
                         title=strExpNme)

# Close if user presses 'cancel':
if objGui.OK is False:
    core.quit()

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
