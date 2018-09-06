# -*- coding: utf-8 -*-
"""
Stimulus presentation for pRF mapping.

The purpose of this script is to present retinotopic mapping stimuli using
Psychopy.
"""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
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

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from itertools import cycle  # import cycle if you want to flicker stimuli
from psychopy.misc import pol2cart
import numpy as np
import pickle
import os
# import scipy.misc
from PIL import Image

# %% Settings for stimulus logging
# If in normal mode, this scrip presents stimuli for population receptive field
# mapping. If in logging mode, this script creates a stimulus log of the
# stimuli used for the pRF mapping that can be used for the pRF finding
# analysis of the py_pRF_mapping library. The stimuli are saved as png files,
# where each png represents the status of visual stimulation for one TR (the
# png files contain modified screenshots of the visual stimulus, and can be
# directly be loaded into the py_pRF_mapping pipepline.

# Logging mode?
lgcLogMde = False

# %% set checkerbar sptial and temporal frequency
# define reversal frequency
tempCyc = 4  # how many bw cycles per s?
# define sptial frequency
spatCyc = 1.5
# set the size of the entire area that the bars should cover [in pixels]
pixCover = 1200  # 1200 is full screen in these settings


# %%
""" SAVING and LOGGING """
# Store info about experiment and experimental run
expName = 'pRF_mapping_log'  # set experiment name here
expInfo = {
    u'participant': u'pilot',
    u'run': u'01',
    }
# Create GUI at the beginning of exp to get more expInfo
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK is False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# get current path and save to variable _thisDir
_thisDir = os.path.dirname(os.path.abspath(__file__))
# get parent path and move up one directory
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
# move to parent_up path
os.chdir(str_path_parent_up)

# Name and create specific subject folder
subjFolderName = str_path_parent_up + os.path.sep + \
    'Log_%s' % (expInfo['participant'])
if not os.path.isdir(subjFolderName):
    os.makedirs(subjFolderName)

# Name and create data folder for the experiment
dataFolderName = subjFolderName + os.path.sep + '%s' % (expInfo['expName'])
if not os.path.isdir(dataFolderName):
    os.makedirs(dataFolderName)

# Name and create specific folder for logging results
logFolderName = dataFolderName + os.path.sep + 'Logging'
if not os.path.isdir(logFolderName):
    os.makedirs(logFolderName)
logFileName = logFolderName + os.path.sep + '%s_%s_Run_%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])

# Name and create specific folder for pickle output
outFolderName = dataFolderName + os.path.sep + 'Output'
if not os.path.isdir(outFolderName):
    os.makedirs(outFolderName)
outFileName = outFolderName + os.path.sep + '%s_%s_Run_%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])

# save a log file and set level for msg to be received
logFile = logging.LogFile(logFileName+'.log', level=logging.INFO)
logging.console.setLevel(logging.WARNING)  # set console to receive warnings


#  %%
"""MONITOR AND WINDOW"""
# set monitor information:
distanceMon = 99  # [99] in scanner
widthMon = 30  # [30] in scanner
PixW = 1920  # [1920.0] in scanner
PixH = 1200  # [1200.0] in scanner

moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab
degCover = misc.pix2deg(pixCover, moni)

# log monitor info
logFile.write('MonitorDistance=' + unicode(distanceMon) + 'cm' + '\n')
logFile.write('MonitorWidth=' + unicode(widthMon) + 'cm' + '\n')
logFile.write('PixelWidth=' + unicode(PixW) + '\n')
logFile.write('PixelHeight=' + unicode(PixH) + '\n')

# set screen:
# for psychoph lab: make 'fullscr = True', set size =(1920, 1080)
myWin = visual.Window(
    size=(PixW, PixH),
    screen=0,
    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
    allowGUI=False,
    allowStencil=True,
    fullscr=True,  # for psychoph lab: fullscr = True
    monitor=moni,
    color=[0, 0, 0],
    colorSpace='rgb',
    units='pix',
    blendMode='avg')

# %%
"""CONDITIONS"""
# retrieve conditions from pickle file (stored in folder Conditions)
str_path_conditions = str_path_parent_up + os.path.sep + 'Conditions' + \
    os.path.sep + 'Conditions_run' + str(expInfo['run']) + '.pickle'
with open(str_path_conditions, 'rb') as handle:
    arrays = pickle.load(handle)
Conditions = arrays["Conditions"]
Conditions = Conditions.astype(int)
TargetOnsetinSec = arrays["TargetOnsetinSec"]
TargetDur = arrays["TargetDuration"]

# If in logging mode, only present stimuli very briefly:
if lgcLogMde:
    # Note: If the 'ExpectedTR' is set too low in logging mode, frames are
    # dropped and the stimuli do not get logged properly.
    ExpectedTR = 0.2
# Otherwise, use actual volume TR:
else:
    ExpectedTR = arrays["TR"]

NrOfSteps = arrays["NrOfSteps"]
NrOfVols = arrays["NrOfVols"]
print('TARGETS: ')
print TargetOnsetinSec

# calculate
Offset = pixCover/NrOfSteps/2
aryOri = [0, 45, 90, 135, 180, 225, 270, 315]
distances = np.linspace(-pixCover/2+Offset, pixCover/2-Offset, NrOfSteps)

aryPosPix = []
for ori in [90, 45, 180, 135, 270, 225, 0, 315]:
    temp = zip(*pol2cart(np.tile(ori, NrOfSteps), distances))
    aryPosPix.append(temp)

# create array to log key pressed events
TriggerPressedArray = np.array([])
TargetPressedArray = np.array([])

logFile.write('Conditions=' + unicode(Conditions) + '\n')
logFile.write('TargetOnsetinSec=' + unicode(TargetOnsetinSec) + '\n')
logFile.write('TargetDur=' + unicode(TargetDur) + '\n')

# %%
"""STIMULI"""

# INITIALISE SOME STIMULI
grating = visual.GratingStim(
    myWin,
    tex="sqrXsqr",
    color=[1.0, 1.0, 1.0],
    colorSpace='rgb',
    opacity=1.0,
    size=(2*PixW, pixCover/NrOfSteps),
    sf=(spatCyc/(pixCover/NrOfSteps), spatCyc/(pixCover/NrOfSteps)),
    ori=0,
    autoLog=False,
    interpolate=False,
    )

# fixation dot
dotFix = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    radius=2,
    fillColor=[1.0, 0.0, 0.0],
    lineColor=[1.0, 0.0, 0.0],)

dotFixSurround = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFixSurround',
    radius=7,
    fillColor=[0.5, 0.5, 0.0],
    lineColor=[0.0, 0.0, 0.0],)

# fixation grid
Circle = visual.Polygon(
    win=myWin,
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
    win=myWin,
    autoLog=False,
    name='Line',
    start=(-PixH, 0),
    end=(PixH, 0),
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
    myWin,
    autoLog=False,
    text='Condition',
    height=30,
    pos=(400, 400)
    )
triggerText = visual.TextStim(
    win=myWin,
    autoLog=False,
    color='white',
    height=30,
    text='Experiment will start soon. \n Waiting for scanner',)
targetText = visual.TextStim(
    win=myWin,
    autoLog=False,
    color='white',
    height=30)

vertices = [(pixCover/2, pixCover/2), (-pixCover/2, pixCover/2),
            (-pixCover/2, -pixCover/2), (pixCover/2, -pixCover/2)]
aperture = visual.Aperture(myWin,
                           autoLog=False,
                           shape=vertices)  # try shape='square'
aperture.enabled = False

# %%
"""TIME AND TIMING PARAMETERS"""

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()  # get screen refresh rate
if refr_rate is not None:
    frameDur = 1.0/round(refr_rate)
else:
    frameDur = 1.0/60.0  # couldn't get a reliable measure so guess
logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# set durations
durations = np.arange(ExpectedTR, ExpectedTR*NrOfVols + ExpectedTR, ExpectedTR)
totalTime = ExpectedTR*NrOfVols

# how many frames b or w? derive from reversal frequency
numFrame = int(round((1/(tempCyc*2))/frameDur))

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
    varCrpY = int(np.around((float(PixH) - float(pixCover)) * 0.5))

    print(('Stimulus log will be cropped by '
           + str(varCrpY)
           + ' in y-direction (screen height).'
           ))

    # Calculate area to crop in x-dimension, at left and right (would be zero
    # is full extend of screeen width was used):
    varCrpX = int(np.around((float(PixW) - float(pixCover)) * 0.5))

    print(('Stimulus log will be cropped by '
           + str(varCrpX)
           + ' in x-direction (screen width).'
           ))

    # Temporary array for screenshots, at full screen size and containing RGB
    # values (needed to obtain buffer content from psychopy):
    aryBuff = np.zeros((PixH, PixW, 3), dtype=np.int16)

    # Prepare array for screenshots. One value per pixel per volume; since the
    # stimuli are greyscale we discard 2nd and 3rd RGB dimension. Also, there
    # is no need to represent the entire screen, just the part of the screen
    # that is actually stimulated (this is typically a square at the centre of
    # the screen, flanked by unstimulated areas on the left and right side).
    aryFrames = np.zeros((pixCover, pixCover, NrOfVols), dtype=np.int16)

    # Make sure that pixCover is of interger type:
    pixCover = int(np.around(pixCover))

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
    myWin.flip()
    event.waitKeys(keyList=['5'], timeStamped=False)

# reset clocks
clock.reset()
logging.data('StartOfRun' + unicode(expInfo['run']))

while clock.getTime() < totalTime:  # noqa

    # get key for motion direction
    keyPos = Conditions[i, 0]
    keyOri = Conditions[i, 1]

    # get direction
    if 0 < keyOri < 9:
        grating.setOpacity(1)
        grating.setOri(aryOri[keyOri-1])
        grating.setPos(aryPosPix[keyOri-1][keyPos])
    else:  # static
        grating.setOpacity(0)

    while clock.getTime() < durations[i]:
        aperture.enabled = True
        # draw fixation grid (circles and lines)
        if not lgcLogMde:
            Circle.setSize((degCover*0.2, degCover*0.2))
            Circle.draw()
            Circle.setSize((degCover*0.4, degCover*0.4))
            Circle.draw()
            Circle.setSize((degCover*0.6, degCover*0.6))
            Circle.draw()
            Circle.setSize((degCover*0.8, degCover*0.8))
            Circle.draw()
            # subtract 0.1 here so that ring is not exactly at outer border
            Circle.setSize((degCover-0.1, degCover-0.1))
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
            (sum(clock.getTime() >= TargetOnsetinSec)
             + sum(clock.getTime() < TargetOnsetinSec + 0.3)
             ) == len(TargetOnsetinSec) + 1):
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
        myWin.flip()

        # handle key presses each frame
        for key in event.getKeys():
            if key in ['escape', 'q']:
                logging.data(msg='User pressed quit')
                myWin.close()
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
              + str(int(NrOfVols))))

        # Temporary array for single frame (3 values per pixel - RGB):
        aryBuff[:, :, :] = myWin.getMovieFrame(buffer='front')

        print(type(aryBuff))
        print('type(aryBuff)')
        print('aryBuff.shape')
        print(aryBuff.shape)

        # We only save one value per pixel per volume (because the stimuli are
        # greyscale we discard 2nd and 3rd RGB dimension):
        aryRgb = aryBuff[:, :, 0]

        # We only save the central square area that contains the stimulus:
        aryFrames[:, :, idxFrame] = \
            np.copy(aryRgb[varCrpY:(varCrpY + pixCover),
                           varCrpX:(varCrpX + pixCover)])
        idxFrame = idxFrame + 1

logging.data('EndOfRun' + unicode(expInfo['run']) + '\n')

# %%
"""TARGET DETECTION RESULTS"""

# calculate target detection results
# create an array 'targetDetected' for showing which targets were detected
targetDetected = np.zeros(len(TargetOnsetinSec))
if len(TargetPressedArray) == 0:
    # if no buttons were pressed
    print "No keys were pressed/registered"
    targetsDet = 0
else:
    # if buttons were pressed:
    for index, target in enumerate(TargetOnsetinSec):
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
              + str(len(TargetOnsetinSec))
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
myWin.flip()
core.wait(5)

# %%
"""CLOSE DISPLAY"""
myWin.close()

# %%
"""SAVE DATA"""

# create python dictionary
output = {'ExperimentName': expInfo['expName'],
          'Date': expInfo['date'],
          'SubjectID': expInfo['participant'],
          'Run_Number': expInfo['run'],
          'Conditions': Conditions,
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
                         + expInfo['run']),
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
    for idxVol in range(NrOfVols):

        print(('---Frame '
              + str(idxVol)
              + ' out of '
              + str(int(NrOfVols))))

        # Create image: TODO: compatibility with new Pillow version?
        im = Image.fromarray(aryFrames[:, :, idxVol])

        # File name (with leading zeros, e.g. '*_004' or '*_042'). For
        # consistency with earlier versions, the numbering of frames (PNG
        # files  corresponding to fMRI volumes) starts at '1' (not at '0').
        strTmpPth = (strPthFrm
                     + os.path.sep
                     + 'run_'
                     + expInfo['run']
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
