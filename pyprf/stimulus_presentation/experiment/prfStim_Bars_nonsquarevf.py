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
import datetime
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from psychopy.misc import pol2cart
from itertools import cycle
from PIL import Image

# Width of monitor [pixels]:
varPixX = 1920  # [1920.0] for 7T scanner
# Height of monitor [pixels]:
varPixY = 1200  # [1200.0] for 7T scanner
strPthNpz = '/home/john/PhD/GitHub/pyprf/pyprf/stimulus_presentation/design_matrices/Run_01.npz'

# Orientation convention is like a clock: 0 is vertical, and positive values rotate clockwise. Beyond 360 and below zero values wrap appropriately.
# NOTE: can/should be float
# lstOri = [0, 45, 90, 135, 180, 225, 270, 315]
# NOTE: WARNING! in the old version, the following list was hard coded at
# some point, perhaps the order of orientation is actually taken from this
# separate, hard-coded list; -- see line 176 of old script
# [90, 45, 180, 135, 270, 225, 0, 315]


def prf_stim(dicParam):
    """
    Present stimuli for population receptive field mapping.

    If in logging mode, this script creates a stimulus log of the stimuli used
    for the pRF mapping that can be used for the pRF finding analysis of the
    pyprf library. The stimuli are saved as png files, where each png
    represents the status of visual stimulation for one TR (the png files
    contain modified screenshots of the visual stimulus, and can be directly be
    loaded into the py_pRF_mapping pipepline.
    """
    # *****************************************************************************
    # *** Experimental parameters (from dictionary)

    # Path of design matrix (npz):
    strPthNpz = dicParam['Path of design matrix (npz)']

    # Output path & file name of log file:
    strPthLog = dicParam['Output path (log files)']

    # Target duration [s]:
    varTrgtDur = dicParam['Target duration [s]']

    # Logging mode (logging mode is for creating files for analysis, not to be
    # used during an experiment).
    lgcLogMde = dicParam['Logging mode']

    # Frequency of stimulus bar in Hz:
    varTmpFrq = dicParam['Temporal frequency [Hz]']

    # Sptial frequency of stimulus (cycles per degree):             ### WARNING CHECK UNITS###
    varSptlFrq = dicParam['Spatial frequency [cyc/deg]']

    # Distance between observer and monitor [cm]:
    varMonDist = dicParam['Distance between observer and monitor [cm]']

    # Width of monitor [cm]:
    varMonWdth = dicParam['Width of monitor [cm]']

    # Width of monitor [pixels]:
    varPixX = dicParam['Width of monitor [pixels]': 1920,]

    # Height of monitor [pixels]:
    varPixY = dicParam['Height of monitor [pixels]']

    # Background colour:
    varBckgrd = dicParam['Background colour [-1 to 1]']

    # *************************************************************************
    # *** Retrieve design matrix

    #        np.savez(strPthNpz,
    #                 aryDsg=aryDsg,
    #                 vecTrgt=vecTrgt,
    #                 lgcFull=lgcFull,
    #                 varTr=varTr,
    #                 varNumVol=varNumVol,
    #                 varNumOri=varNumOri,
    #                 varNumPos=varNumPos,
    #                 varNumTrgt=varNumTrgt,
    #                 varIti=varIti)    

    # Load stimulus parameters from npz file.
    objNpz = np.load(strPthNpz)

    # Get design matrix (for bar positions and orientation):
    aryDsg = objNpz['aryDsg']

    # Vector with times of target events:
    vecTrgt = objNpz['vecTrgt']

    # Full screen mode? If no, bar stimuli are restricted to a central square.
    # If yes, bars appear on the entire screen. This parameter is set when
    # creating the design matrix.
    lgcFull = bool(objNpz['lgcFull'])

    # Number of volumes:
    # varNumVol = int(objNpz['varNumVol'])

    # Number of bar positions:
    # varNumPos = int(objNpz['varNumPos'])

    # Number of orientation:
    # varNumOri = int(objNpz['varNumOri'])

    # Number of target events:
    # varNumTrgt = int(objNpz['varNumTrgt'])

    # Average inter-trial interval for target events:
    # varIti = float(objNpz['varIti'])

    # If in logging mode, only present stimuli very briefly:
    if lgcLogMde:
        # Note: If 'varTr' is set too low in logging mode, frames are
        # dropped and the stimuli do not get logged properly.
        varTr = 0.2
    # Otherwise, use actual volume TR:
    else:
        # Volume TR:
        varTr = float(objNpz['varTr'])

    # *************************************************************************
    # *** Logging

    # Set clock:
    objClck = core.Clock()

    # Set clock for logging:
    logging.setDefaultClock(objClck)

    # Create a log file and set logging verbosity:
    fleLog = logging.LogFile(strPthLog, level=logging.DATA)

    # Log stimulus parameters:
    fleLog.write('Log file path: ' + strPthLog + '\n')
    fleLog.write('Design matrix: ' + strPthNpz + '\n')
    fleLog.write('Full screen: ' + str(lgcFull) + '\n')
    fleLog.write('Volume TR [s]: ' + str(varTr) + '\n')
    fleLog.write('Frequency of stimulus bar in Hz: ' + str(varTmpFrq) + '\n')
    fleLog.write('Sptial frequency of stimulus (cycles per degree): '
                 + str(varSptlFrq) + '\n')
    fleLog.write('Distance between observer and monitor [cm]: '
                 + str(varMonDist) + '\n')
    fleLog.write('Width of monitor [cm]: ' + str(varMonWdth) + '\n')
    fleLog.write('Width of monitor [pixels]: ' + str(varPixX) + '\n')
    fleLog.write('Height of monitor [pixels]: ' + str(varPixY) + '\n')
    fleLog.write('Background colour [-1 to 1]: ' + str(varBckgrd) + '\n')
    fleLog.write('Target duration [s]: ' + str(varTrgtDur) + '\n')
    fleLog.write('Logging mode: ' + str(lgcLogMde) + '\n')

    # Set console logging verbosity:
    logging.console.setLevel(logging.WARNING)

    # *************************************************************************
    # *** Prepare behavioural response logging

    # Array for logging of key presses:
    aryKeys = np.array([], dtype=np.float32)

    # Control the logging of participant responses:
    varSwtRspLog = 0

    # The key that the participant has to press after a target event:
    strTrgtKey = '1'

    # Counter for correct/incorrect responses:
    varCntHit = 0  # Counter for hits
    varCntMis = 0  # Counter for misses

    # *************************************************************************
    # *** Setup

    # Create monitor object:
    objMon = monitors.Monitor('Screen_7T_NOVA_32_Channel_Coil',
                              width=varMonWdth,
                              distance=varMonDist)

    # Set size of monitor:
    objMon.setSizePix([varPixX, varPixY])

    # Set screen:
    objWin = visual.Window(
        size=(varPixX, varPixY),
        screen=0,
        winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
        allowGUI=False,
        allowStencil=True,
        fullscr=True,
        monitor=objMon,
        color=varBckgrd,
        colorSpace='rgb',
        units='deg',
        blendMode='avg'
        )

    # *************************************************************************
    # *** Prepare stimulus properties

    # The area that will be covered by the bar stimulus depends on whether
    # presenting in full screen mode or not. If in full screen mode, the
    # entire width of the screen will be covered. If not, a central square
    # with a side length equal to the screen height will be covered.
    if lgcFull:
        varPixCov = varPixX
    else:
        varPixCov = varPixY

    # Convert size in pixels to size in degrees (given the monitor settings):
    varDegCover = misc.pix2deg(varPixCov, objMon)

    # The thickness of the bar stimulus depends on the size of the screen to
    # be covered, and on the number of positions at which to present the bar.
    # Bar thickness in pixels:
    varThckPix = float(varPixCov) / float(varNumPos)
    # Bar thickness in degree:
    varThckDgr = misc.pix2deg(varThckPix, objMon)

    # Offset of the bar stimuli. The bar stimuli should cover the screen area,
    # without extending beyond the screen. Because their position refers to
    # the centre of the bar, we need to limit the extend of positions at the
    # edge of the screen by an offset, Offset in pixels:
    varOffsetPix = varThckPix / 2.0

    # Maximum bar position in pixels, with respect to origin at centre of
    # screen:
    varPosMaxPix = float(varPixCov) / 2.0 - float(varOffsetPix)

    # Array of possible bar positions (displacement relative to origin at
    # centre of the screen) in pixels:
    vecPosPix = np.linspace((-varPosMaxPix), varPosMaxPix, varNumPos)

    # Vectors with as repititions of orientations and positions, so that there
    # is one unique combination of position & orientation for all positions
    # and orientations:
    vecOriRep = np.repeat(lstOri, varNumPos)
    vecPosPixRep = np.tile(vecPosPix, varNumOri)

    # Convert from polar to cartesian coordinates.
    # theta, radius = pol2cart(x, y, units=’deg’)
    vecTheta, vecRadius = pol2cart(vecOriRep, vecPosPixRep, units='deg')

    # lstPosPix = []
    # for ori in lstOri:
    #     temp = zip(*pol2cart(np.tile(ori, varNumPos), vecPosPix))
    #     lstPosPix.append(temp)

    # Bar stimulus size (length & thickness), in pixels:
    tplBarSzePix = (int(varPixCov), int(varThckPix))
    
    # Bar stimulus spatial frequency (in x & y directions):
    tplBarSf = (float(varSptlFrq) / varThckPix,
                float(varSptlFrq) / varThckPix)
   
    # *************************************************************************
    # *** Stimuli

    # TODO: varPixCov = varPix

    # Bar stimulus:
    objBar = visual.GratingStim(
        objWin,
        tex='sqrXsqr',
        color=[1.0, 1.0, 1.0],
        colorSpace='rgb',
        opacity=1.0,
        size=tplBarSzePix,
        sf=tplBarSf,
        ori=0.0,
        autoLog=False,
        interpolate=False,
        units='pix'                           ##############CHECK###################
        )

    # Fixation dot:
    objFixDot = visual.Circle(
        objWin,
        radius=2.0,
        fillColor=[1.0, 0.0, 0.0],
        lineColor=[1.0, 0.0, 0.0],
        autoLog=False,
        units='pix')                           ##############CHECK###################

    # Fixation dot surround:
    objFixSrr = visual.Circle(
        objWin,
        radius=7.0,
        fillColor=[0.5, 0.5, 0.0],
        lineColor=[0.0, 0.0, 0.0],
        autoLog=False,
        units='pix')                           ##############CHECK###################

    # Fixation grid circle:
    objGrdCrcl = visual.Polygon(
        win=objWin,
        edges=90,
        ori=0.0,
        pos=[0, 0],
        lineWidth=2,
        lineColor=[1.0, 1.0, 1.0],
        lineColorSpace='rgb',
        fillColor=None,
        fillColorSpace='rgb',
        opacity=1.0,
        autoLog=False,
        interpolate=True,
        units='deg')

    # Fixation grid line:
    objGrdLne = visual.Line(
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

    if not(lgcFull):
        # List with aperture coordinates (used to cover left and right side of
        # the screen when not in full-screen mode). List of tuples.
        lstAptrCor = [(varPixCov / 2, varPixCov / 2),
                      (-varPixCov / 2, varPixCov / 2),
                      (-varPixCov / 2, -varPixCov / 2),
                      (varPixCov / 2, -varPixCov / 2)] 

        # Aperture for covering left and right side of screen if not stimulating full screen.
        objAprtr = visual.Aperture(objWin,
                                   autoLog=False,
                                   shape=lstAptrCor)
        objAprtr.enabled = False

    # *************************************************************************
    # *** Timing

    # Get screen refresh rate
    varFps = objWin.getActualFrameRate()
    if varFps is not None:
        varFrmeDur = 1.0 / float(varFps)
    else:
        # Guessing refresh rate:
        varFrmeDur = 1.0 / 60.0
    fleLog.write('Frames per second: ' + str(varFps) + '\n')

    # set durations
    durations = np.arange(varTr, varTr*varNumVol + varTr, varTr)

    totalTime = varTr*varNumVol

    # how many frames b or w? derive from reversal frequency
    numFrame = int(round((1/(varTmpFrq*2))/varFrmeDur))

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
            objBar.setOpacity(1)
            objBar.setOri(lstOri[keyOri-1])
            objBar.setPos(lstPosPix[keyOri-1][keyPos])
        else:  # static
            objBar.setOpacity(0)

        while clock.getTime() < durations[i]:
            # objAprtr.enabled = True
            # draw fixation grid (objGrdCrcls and lines)
            if not lgcLogMde:
                objGrdCrcl.setSize((varDegCover*0.2, varDegCover*0.2))
                objGrdCrcl.draw()
                objGrdCrcl.setSize((varDegCover*0.4, varDegCover*0.4))
                objGrdCrcl.draw()
                objGrdCrcl.setSize((varDegCover*0.6, varDegCover*0.6))
                objGrdCrcl.draw()
                objGrdCrcl.setSize((varDegCover*0.8, varDegCover*0.8))
                objGrdCrcl.draw()
                # subtract 0.1 here so that ring is not exactly at outer border
                objGrdCrcl.setSize((varDegCover-0.1, varDegCover-0.1))
                objGrdCrcl.draw()
                objGrdLne.setOri(0)
                objGrdLne.draw()
                objGrdLne.setOri(45)
                objGrdLne.draw()
                objGrdLne.setOri(90)
                objGrdLne.draw()
                objGrdLne.setOri(135)
                objGrdLne.draw()

            # update contrast flicker with square wave
            y = squFlicker()
            objBar.contrast = y

            objBar.draw()
            objAprtr.enabled = False

            # decide whether to draw target
            # first time in target interval? reset target counter to 0!
            if (
                (sum(clock.getTime() >= vecTrgt)
                 + sum(clock.getTime() < vecTrgt + 0.3)
                 ) == len(vecTrgt) + 1):
                # display target!
                # change color fix dot surround to red
                objFixSrr.fillColor = [0.5, 0.0, 0.0]
                objFixSrr.lineColor = [0.5, 0.0, 0.0]
            # dont display target!
            else:
                # keep color fix dot surround yellow
                objFixSrr.fillColor = [0.5, 0.5, 0.0]
                objFixSrr.lineColor = [0.5, 0.5, 0.0]

            if not lgcLogMde:
                # draw fixation point surround
                objFixSrr.draw()
                # draw fixation point
                objFixDot.draw()

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
    fleLog.write(unicode(resultText) + '\n')
    fleLog.write(unicode(feedbackText) + '\n')
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

    # *************************************************************************
    # *** Stimulus parameters



    # *************************************************************************
    # *** GUI 

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
    dicParam = {'Run (name of design matrix file)': 'Run_01',
                'Target duration [s]': 0.3,
                'Logging mode': [False, True],
                'Temporal frequency [Hz]': 4.0,
                'Spatial frequency [cyc/deg]': 1.5,    ### WARNING CHECK UNITS###
                'Distance between observer and monitor [cm]': 99.0,
                'Width of monitor [cm]': 30.0,
                'Width of monitor [pixels]': 1920,
                'Height of monitor [pixels]': 1200,
                'Background colour [-1 to 1]': 0.7}

    if not(strFleNme is None):

        # If an input file name is provided, put it into the dictionary (as
        # default, can still be overwritten by user).
        dicParam['Output file name'] = strFleNme

    if strGui == 'True':

        # Pop-up GUI to let the user select parameters:
        objGui = gui.DlgFromDict(dictionary=dicParam,
                                 title='Design Matrix Parameters')

    # Start experiment if user presses 'ok':
    if objGui.OK is True:

        # Get date string as default session name:
        strDate = str(datetime.datetime.now())
        lstDate = strDate[0:10].split('-')
        lstTime = strDate[11:19].split(':')
        strDate = (lstDate[0] + lstDate[1] + lstDate[2] + '_' + lstTime[0]
                   + lstTime[1] + lstTime[2])

        # Path of parent directory:
        strPth = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
        # Output path ('~/pyprf/pyprf/stimulus_presentation/design_matrices/'):
        strPthOut = os.path.join(strPth,
                                'log',
                                (dicParam['Output file name']
                                 + strDate
                                 + '.txt')
                                )
    
        # Add output path to dictionary.
        dicParam['Output path (log files)'] = strPthOut

        # Path of design matrix file (npz):
        strPthNpz = os.path.join(strPth,
                                'design_matrices',
                                (dicParam['Run (name of design matrix file)']
                                 + '.npz')
                                )
    
        # Add path of design matrix (npz file) to dictionary.
        dicParam['Path of design matrix (npz)'] = strPthNpz
    
        prf_stim(dicParam)

    else:

        # Close GUI if user presses 'cancel':
        core.quit()
