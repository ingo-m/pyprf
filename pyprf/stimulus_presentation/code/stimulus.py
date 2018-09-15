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
import argparse
import numpy as np
import datetime
from psychopy import visual, event, core,  monitors, logging, gui
from psychopy.tools.monitorunittools import pix2deg

# strPthNpz = '/home/john/PhD/GitHub/pyprf/pyprf/stimulus_presentation/design_matrices/Run_01.npz'


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
    varTrgtDur = float(dicParam['Target duration [s]'])

    # Logging mode (logging mode is for creating files for analysis, not to be
    # used during an experiment).
    lgcLogMde = dicParam['Logging mode']

    # Directory where to save stimulus log (frames) for analysis if in logging
    # mode.
    strPthFrm = dicParam['Output path stimulus log (frames)']

    # Frequency of stimulus bar in Hz:
    varGrtFrq = float(dicParam['Temporal frequency [Hz]'])

    # Sptial frequency of stimulus (cycles along width of bar stimulus):
    varBarSf = float(dicParam['Spatial frequency [cyc per bar]'])

    # Distance between observer and monitor [cm]:
    varMonDist = float(dicParam['Distance between observer and monitor [cm]'])

    # Width of monitor [cm]:
    varMonWdth = float(dicParam['Width of monitor [cm]'])

    # Width of monitor [pixels]:
    varPixX = int(dicParam['Width of monitor [pixels]'])

    # Height of monitor [pixels]:
    varPixY = int(dicParam['Height of monitor [pixels]'])

    # Background colour:
    varBckgrd = float(dicParam['Background colour [-1 to 1]'])

    # Show fixation grid?
    lgcGrd = dicParam['Show fixation grid?']

    # *************************************************************************
    # *** Retrieve design matrix

    # Load stimulus parameters from npz file.
    objNpz = np.load(strPthNpz)

    # Get design matrix (for bar positions and orientation):
    aryDsg = objNpz['aryDsg']

    # Number of volumes:
    varNumVol = aryDsg.shape[0]

    # Vector with times of target events:
    vecTrgt = objNpz['vecTrgt']

    # Number of target events:
    varNumTrgt = vecTrgt.shape[0]

    # Full screen mode? If no, bar stimuli are restricted to a central square.
    # If yes, bars appear on the entire screen. This parameter is set when
    # creating the design matrix.
    lgcFull = bool(objNpz['lgcFull'])

    # If in logging mode, only present stimuli very briefly:
    if lgcLogMde:

        # Conditional import:
        from PIL import Image

        # Note: If 'varTr' is set too low in logging mode, frames are
        # dropped and the stimuli do not get logged properly.
        varTr = 0.2

        # In log mode, don't show grid.
        lgcGrd = False

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
    fleLog.write('Volume TR [s] (from design matrix): ' + str(varTr) + '\n')
    fleLog.write('Frequency of stimulus bar in Hz: ' + str(varGrtFrq) + '\n')
    fleLog.write('Sptial frequency of stimulus (cycles along width of '
                 + 'bar stimulus): ' + str(varBarSf) + '\n')
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

    # Switch target (show target or not?):
    varSwtTrgt = 0

    # Control the logging of participant responses:
    varSwtRspLog = 0

    # The key that the participant has to press after a target event:
    strTrgtKey = '1'

    # Counter for correct/incorrect responses:
    varCntHit = 0  # Counter for hits
    varCntMis = 0  # Counter for misses

    # Time (in seconds) that participants have to respond to a target event in
    # order for the event to be logged as a hit:
    varHitTme = 2.0

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
        winType='pyglet',  # winType : None, 'pyglet', 'pygame'
        allowGUI=False,
        allowStencil=True,
        fullscr=True,
        monitor=objMon,
        color=varBckgrd,
        colorSpace='rgb',
        units='deg',
        blendMode='avg')

    # *************************************************************************
    # *** Spatial stimulus properties

    # The area that will be covered by the bar stimulus depends on whether
    # presenting in full screen mode or not. If in full screen mode, the
    # entire width of the screen will be covered. If not, a central square
    # with a side length equal to the screen height will be covered.
    if lgcFull:
        varPixCov = varPixX
    else:
        varPixCov = varPixY

    # Convert size in pixels to size in degrees (given the monitor settings):
    # varDegCover = pix2deg(varPixCov, objMon)

    # Numberic codes used for bar positions:
    vecPosCode = np.unique(aryDsg[:, 1])

    # Number of bar positions:
    varNumPos = vecPosCode.shape[0]

    # The thickness of the bar stimulus depends on the size of the screen to
    # be covered, and on the number of positions at which to present the bar.
    # Bar thickness in pixels:
    varThckPix = float(varPixCov) / float(varNumPos)

    # Bar thickness in degree:
    varThckDgr = np.around(pix2deg(varThckPix, objMon), decimals=5)

    # Write stimulus parameters to log file.
    fleLog.write('* * * Stimulus properties in degrees of visual angle '
                 + '* * * \n')
    fleLog.write('Width of bar stimulus [deg]: ' + str(varThckDgr) + '\n')

    # Spatial frequency of bar stimulus is defined (in user input) as cycles
    # along width of the bar  stimulus. We need to convert this to cycles per
    # pixel (for the stimulus creation) and to cycles per degree (for
    # reference, written to log file).

    # Spatial frequency in cycles per pixel:
    varBarSfPix = float(varBarSf) / varThckPix
    tplBarSfPix = (varBarSfPix, varBarSfPix)

    # Spatial frequency in cycles per degree of visual angle (for reference
    # only):
    varBarSfDeg = np.around((float(varBarSf) / varThckDgr), decimals=5)

    # Write stimulus parameters to log file.
    fleLog.write('Spatial frequency of bar stimulus [cyc/deg]: '
                 + str(varBarSfDeg) + '\n')
    fleLog.write('* * * \n')

    # Bar stimulus size (length & thickness), in pixels.
    tplBarSzePix = (varPixX, int(varThckPix))

    # Offset of the bar stimuli. The bar stimuli should cover the screen area,
    # without extending beyond the screen. Because their position refers to
    # the centre of the bar, we need to limit the extend of positions at the
    # edge of the screen by an offset, Offset in pixels:
    varOffsetPix = varThckPix * 0.5

    # Maximum bar position in pixels, with respect to origin at centre of
    # screen:
    varPosMaxPix = (float(varPixCov) * 0.5) - float(varOffsetPix)

    # Array of possible bar positions (displacement relative to origin at
    # centre of the screen) in pixels:
    vecPosPix = np.linspace(-varPosMaxPix, varPosMaxPix, varNumPos,
                            endpoint=True)

    # Replace numeric position codes with pixel position values:
    for idxPos, varPos in enumerate(vecPosCode):

        # Replace current position code, if this is not a rest block:
        vecLgc = np.multiply((aryDsg[:, 1] == varPos),
                             (aryDsg[:, 0] != 0.0))

        # Place pixel position value in design matrix:
        aryDsg[vecLgc, 1] = vecPosPix[idxPos]

    # Psychopy orientation convention: "Orientation convention is like a clock:
    # 0 is vertical, and positive values rotate clockwise." Actually, 0 is the
    # positive x-axis. Orientations are coded as follows: horizontal = 0.0,
    # vertical = 90.0, lower left to upper right = 45.0, upper left to lower
    # right = 135.0. We need to convert psychopy orientation & direction
    # convention into x and y coordinates.
    lstPos = [None] * varNumVol
    for idxVol in range(varNumVol):

        # Get angle and radius of current volume:
        varRad = float(aryDsg[idxVol, 1])
        varAngle = float(aryDsg[idxVol, 2])

        # Horizontal:
        if varAngle == 0.0:
            varTmpX = 0.0
            varTmpY = varRad

        # Vertical:
        elif varAngle == 90.0:
            varTmpX = varRad
            varTmpY = 0.0

        # Lower left to upper right:
        elif varAngle == 45.0:
            if varRad < 0.0:
                varTmpX = -np.sqrt(
                                   np.add(
                                          np.square(varRad),
                                          np.square(varRad)
                                          )
                                   )
            elif 0.0 < varRad:
                varTmpX = np.sqrt(
                                  np.add(
                                         np.square(varRad),
                                         np.square(varRad)
                                         )
                                  )
            else:
                varTmpX = 0.0
            varTmpY = 0.0

        # Upper left to lower right:
        elif varAngle == 135.0:
            if varRad < 0.0:
                varTmpX = -np.sqrt(
                                   np.add(
                                          np.square(varRad),
                                          np.square(varRad)
                                          )
                                   )
            elif 0.0 < varRad:
                varTmpX = np.sqrt(
                                  np.add(
                                         np.square(varRad),
                                         np.square(varRad)
                                         )
                                  )
            else:
                varTmpX = 0.0
            varTmpY = 0.0

        # Position is coded as a tuple:
        lstPos[idxVol] = (varTmpX, varTmpY)

    # *************************************************************************
    # *** Stimuli

    # Bar stimulus:
    objBar = visual.GratingStim(
        objWin,
        contrast=1.0,
        pos=(0.0, 0.0),
        tex='sqrXsqr',
        color=[1.0, 1.0, 1.0],
        colorSpace='rgb',
        opacity=1.0,
        size=tplBarSzePix,
        sf=tplBarSfPix,
        ori=0.0,
        autoLog=False,
        interpolate=False,
        units='pix')

    # Colour of fixation dot:
    lstClrFix = [-0.69, 0.83, 0.63]
    # lstClrFix = [0.04, 0.95, -1.0]

    # Colour of fixation dot when it becomes a target:
    lstClrTrgt = [0.95, 0.04, -1.0]

    # Fixation dot:
    objFix = visual.Circle(
        objWin,
        units='deg',
        pos=(0.0, 0.0),
        radius=0.05,
        edges=24,
        fillColor=lstClrFix,
        fillColorSpace='rgb',
        lineColor=lstClrFix,
        lineColorSpace='rgb',
        lineWidth=0.0,
        interpolate=False,
        autoLog=False)

    # Fication dot surround:
    objFixSrd = visual.Circle(
        objWin,
        units='deg',
        pos=(0.0, 0.0),
        radius=0.09,
        edges=24,
        fillColor=lstClrTrgt,
        fillColorSpace='rgb',
        lineColor=lstClrTrgt,
        lineColorSpace='rgb',
        lineWidth=0.0,
        interpolate=False,
        autoLog=False)

    if lgcGrd:

        # Number of grid circles:
        varNumCrcl = 3

        # Radi at which to present grid circles:
        vecGrdCrclRad = np.linspace((0.25 * float(varPixY)),
                                    (0.75 * float(varPixY)),
                                    num=varNumCrcl)

        # In practice 'radius' seems to refer to refer to the diameter of the
        # circle.

        # Fixation grid circles:
        lstGrdCrcl = [None] * varNumCrcl
        for idxCrcl, varRad in enumerate(vecGrdCrclRad):
            lstGrdCrcl[idxCrcl] = visual.Circle(
                win=objWin,
                pos=(0.0, 0.0),
                radius=varRad,
                edges=128,
                lineWidth=2.0,
                lineColor=[1.0, 1.0, 1.0],
                lineColorSpace='rgb',
                fillColor=None,
                fillColorSpace='rgb',
                opacity=1.0,
                autoLog=False,
                interpolate=True,
                units='pix')

        # Fixation grid line:
        lstGrdLne = [None] * 4
        for idxLne, varOri in enumerate([0.0, 45.0, 90.0, 135.0]):
            lstGrdLne[idxLne] = visual.Line(
                win=objWin,
                ori=varOri,
                start=(int(-varPixY), 0),
                end=(int(varPixY), 0),
                pos=(0.0, 0.0),
                lineWidth=2.0,
                lineColor=[1.0, 1.0, 1.0],
                lineColorSpace='rgb',
                fillColor=None,
                fillColorSpace='rgb',
                opacity=1.0,
                autoLog=False,
                interpolate=True,
                units='pix')

    # *************************************************************************
    # *** Aperture

    # The area that will be covered by the bar stimulus depends on whether
    # presenting in full screen mode or not. If in full screen mode, the entire
    # width of the screen will be covered. If not, a central square with a side
    # length equal to the screen height will be covered.

    if not(lgcFull):

        # Aperture side length in degree. For some reason, the aperture does
        # not seem to accept pixel units.
        varDegCov = pix2deg(float(varPixCov), objMon)

        # Aperture for covering left and right side of screen if not presenting
        # in full screen mode.
        objAprtr = visual.Aperture(
            objWin,
            size=varDegCov,
            pos=(0, 0),
            shape='square',
            inverted=False,
            units='deg')

        objAprtr.enabled = True

    # *************************************************************************
    # *** Logging mode preparations

    if lgcLogMde:

        print('Logging mode')

        # Calculate area to crop in x-dimension, at left and right (zero is
        # full extend of screeen width is used):
        if lgcFull:
            varCrpX = 0
        else:
            varCrpX = int(np.around((float(varPixX) - float(varPixY)) * 0.5))

        print(('Stimulus log will be cropped by '
               + str(varCrpX)
               + ' in x-direction (screen width).'
               ))

        # Temporary array for screenshots, at full screen size and containing
        # RGB values (needed to obtain buffer content from psychopy):
        aryBuff = np.zeros((varPixY, varPixX, 3), dtype=np.int8)

        # It is not necessary to sample every pixel; only every second pixel is
        # sampled. Number of pixel to be sampled along x and y direction:
        varHalfPixX = int(np.around(varPixX * 0.5))
        varHalfPixY = int(np.around(varPixY * 0.5))

        # Prepare array for screenshots. One value per pixel per volume; since
        # the stimuli are greyscale we discard 2nd and 3rd RGB dimension. Also,
        # there is no need to represent the entire screen, just the part of the
        # screen that is actually stimulated (if not in full screen mode, this
        # is a square at the centre of the screen, flanked by unstimulated
        # areas on the left and right side).
        if lgcFull:
            aryFrames = np.zeros((varHalfPixY,
                                  varHalfPixX,
                                  varNumVol), dtype=np.int8)
        else:
            aryFrames = np.zeros((varHalfPixY,
                                  varHalfPixY,
                                  varNumVol), dtype=np.int8)

        # Counter for screenshots:
        idxFrame = 0

    # *************************************************************************
    # *** Timing & switches

    # Target counter:
    varCntTrgt = 0

    # Time of the first target event:
    varTmeTrgt = vecTrgt[varCntTrgt]

    # Switch for grating polarity flicker:
    varSwtGrt = 0

    # The input parameter 'varGrtFrq' gives the grating flicker frequency in
    # Hz. We need to convert to second:
    varGrtDur = 1.0 / float(varGrtFrq)

    # *************************************************************************
    # *** Presentation

    # Hide the mouse cursor:
    event.Mouse(visible=False)

    if not(lgcLogMde):

        if lgcGrd:

            # Draw fixation grid circles:
            for objGrdCrcl in lstGrdCrcl:
                objGrdCrcl.draw(win=objWin)

            # Draw fixation grid lines:
            for objGrdLne in lstGrdLne:
                objGrdLne.draw(win=objWin)

        # Draw fixation dot & surround:
        objFixSrd.draw(win=objWin)
        objFix.draw(win=objWin)

        objWin.flip()

        # Wait for scanner trigger pulse & set clock after receiving trigger
        # pulse (scanner trigger pulse is received as button press ('5')):
        strTrgr = ['0']
        while strTrgr[0][0] != '5':
            # Check for keypress:
            lstTmp = event.getKeys(keyList=['5'], timeStamped=False)
            # Whether the list has the correct length (if nothing has happened,
        # lstTmp # will have length zero):
            if len(lstTmp) == 1:
                strTrgr = lstTmp[0][0]

    # Trigger pulse received, reset clock:
    objClck.reset(newT=0.0)

    # Main timer which represents the starting point of the experiment:
    varTme01 = objClck.getTime()

    # Time that is updated continuously to track time:
    varTme02 = objClck.getTime()

    # Timer used to control the logging of stimulus events:
    varTme03 = objClck.getTime()

    # Timer for grating stimulus polarity flicker:
    varTme04 = objClck.getTime()
    varTme05 = objClck.getTime()

    # Start of the experiment:
    for idxVol in range(varNumVol):

        # Show a grating during this volume?
        lgcOn = (aryDsg[idxVol, 0] == 1.0)

        # Set grating properties for current volume:
        if lgcOn:

            # Get stimulus properties from design matrix:
            varTmpPos = lstPos[idxVol]
            varTmpOri = aryDsg[idxVol, 2]
            varTmpCon = aryDsg[idxVol, 3]

            # Set bar properties:
            objBar.setPos(varTmpPos)
            objBar.setOri(varTmpOri)
            objBar.setColor((varTmpCon, varTmpCon, varTmpCon))

        # Still on the same volume?
        while varTme02 < (varTme01 + (float(idxVol + 1) * varTr)):

            # *****************************************************************
            # *** Draw stimuli

            # Draw fixation grid?
            if lgcGrd:

                # Draw fixation grid circles:
                for objGrdCrcl in lstGrdCrcl:
                    objGrdCrcl.draw(win=objWin)

                # Draw fixation grid lines:
                for objGrdLne in lstGrdLne:
                    objGrdLne.draw(win=objWin)

            # If a grating is shown, which orientation, position, and contrast?
            if lgcOn:

                # Draw grating.
                objBar.draw(win=objWin)

            # Draw fixation dot & surround:
            objFixSrd.draw(win=objWin)
            objFix.draw(win=objWin)

            # Flip drawn objects to screen:
            objWin.flip()

            # Update current time:
            varTme02 = objClck.getTime()

            # Update current time:
            varTme05 = objClck.getTime()

            # *****************************************************************
            # *** Target control

            # Time for target?
            if ((varTmeTrgt <= varTme02)
                    and (varTme02 <= (varTmeTrgt + varTrgtDur))):

                # Was the target off on the previous frame?
                if varSwtTrgt == 0:

                    # Switch the target on by changing the fixation dot colour.
                    objFix.fillColor = lstClrTrgt

                    # Log target event:
                    strTmp = ('TARGET scheduled for: '
                              + str(varTmeTrgt))
                    logging.data(strTmp)

                    # Once after target onset we set varSwtRspLog to one so
                    # that the participant's respond can be logged:
                    varSwtRspLog = 1

                    # Likewise, just after target onset we set the timer for
                    # response logging to the current time so that the response
                    # will only be counted as a hit in a specified time
                    # interval after target onset:
                    varTme03 = objClck.getTime()

                    # Switch the target switch.
                    varSwtTrgt = 1

            else:

                # No time for target.

                # Was the target just on?
                if varSwtTrgt == 1:

                    # Switch the target off (by changing fixation dot colour
                    # back to normal).
                    objFix.fillColor = lstClrFix

                    # Switch the target switch.
                    varSwtTrgt = 0

                    # Only increase the target  counter if the last target has
                    # not been reached yet:
                    if (varCntTrgt + 1) < varNumTrgt:

                        # Increase the target counter:
                        varCntTrgt = varCntTrgt + 1

                        # Time of next target event:
                        varTmeTrgt = vecTrgt[varCntTrgt]

            # Has the participant's response not been reported yet, and is it
            # still within the time window?
            if (varSwtRspLog == 1) and (varTme02 <= (varTme03 + varHitTme)):

                # Check for and log participant's response:
                lstRsps = event.getKeys(keyList=[strTrgtKey],
                                        timeStamped=False)

                # Check whether the list has the correct length:
                if len(lstRsps) == 1:

                    # Does the list contain the response key?
                    if lstRsps[0] == strTrgtKey:

                        # Log hit:
                        logging.data('Hit')

                        # Count hit:
                        varCntHit += 1

                        # After logging the hit, we have to switch off the
                        # response logging, so that the same hit is not logged
                        # over and over again:
                        varSwtRspLog = 0

            elif (varSwtRspLog == 1) and (varTme02 > (varTme03 + varHitTme)):

                # Log miss:
                logging.data('Miss')

                # Count miss:
                varCntMis += 1

                # If the subject does not respond to the target within time, we
                # log this as a miss and set varSwtRspLog to zero (so that the
                # response won't be logged as a hit anymore afterwards):
                varSwtRspLog = 0

            # *****************************************************************
            # *** Grating control

            # If a grating is shown, which orientation, position, and contrast?
            if lgcOn:

                # Change grating polarity:
                if (varTme04 + varGrtDur) <= varTme05:

                    if varSwtGrt == 0:
                        varSwtGrt = 1
                        objBar.contrast = 1.0

                    else:
                        varSwtGrt = 0
                        objBar.contrast = -1.0

                    # Remember time at which grating polarity was switched:
                    varTme04 = objClck.getTime()

            # Update current time:
            # varTme02 = objClck.getTime()

            # Update current time:
            # varTme05 = objClck.getTime()

        if lgcLogMde:

            # Temporary array for single frame (3 values per pixel - RGB):
            aryBuff[:, :, :] = objWin.getMovieFrame(buffer='front')

            # We only save one value per pixel per volume (because the stimuli
            # are greyscale we discard 2nd and 3rd RGB dimension):
            aryRgb = aryBuff[:, :, 0]

            # Sample the relevant part of the screen (all the screen if in
            # full screen mode, central square otherwise).0
            aryRgb = aryRgb[:, varCrpX:(varPixX - varCrpX)].astype(np.int8)

            # Only sample every second pixel:
            aryRgb = aryRgb[::2, ::2]

            aryFrames[:, :, idxFrame] = np.copy(aryRgb)

            idxFrame = idxFrame + 1

        # Check whether exit keys have been pressed:
        if func_exit() == 1:
            break

    print('echo1')

    # *************************************************************************
    # *** Feedback

    logging.data('------End of the experiment.------')

    # Performance feedback only if there were any targets:
    if 0.0 < float(varCntHit + varCntMis):

        # Ratio of hits:
        varHitRatio = float(varCntHit) / float(varCntHit + varCntMis)

        # Present participant with feedback on her target detection
        # performance:
        if 0.99 < varHitRatio:
            # Perfect performance:
            strFeedback = ('You have detected '
                           + str(varCntHit)
                           + ' targets out of '
                           + str(varCntHit + varCntMis)
                           + '\n'
                           + 'Keep up the good work :)')
        elif 0.9 < varHitRatio:
            # OKish performance:
            strFeedback = ('You have detected '
                           + str(varCntHit)
                           + ' targets out of '
                           + str(varCntHit + varCntMis)
                           + '\n'
                           + 'There is still room for improvement.')
        else:
            # Low performance:
            strFeedback = ('You have detected '
                           + str(varCntHit)
                           + ' targets out of '
                           + str(varCntHit + varCntMis)
                           + '\n'
                           + 'Please try to focus more.')

        # Create text object:
        objTxtTmr = visual.TextStim(objWin,
                                    text=strFeedback,
                                    font="Courier New",
                                    pos=(0.0, 0.0),
                                    color=(1.0, 1.0, 1.0),
                                    colorSpace='rgb',
                                    opacity=1.0,
                                    contrast=1.0,
                                    ori=0.0,
                                    height=0.5,
                                    antialias=True,
                                    alignHoriz='center',
                                    alignVert='center',
                                    flipHoriz=False,
                                    flipVert=False,
                                    autoLog=False)

        # Show feedback text:
        varTme04 = objClck.getTime()
        while varTme02 < (varTme04 + 3.0):
            objTxtTmr.draw()
            objWin.flip()
            varTme02 = objClck.getTime()

        # Log total number of hits and misses:
        logging.data(('Number of hits: ' + str(varCntHit)))
        logging.data(('Number of misses: ' + str(varCntMis)))
        logging.data(('Percentage of hits: '
                      + str(np.around((varHitRatio * 100.0), decimals=1))))

    print('echo2')

    # *************************************************************************
    # *** Logging mode

    print('echo3')

    # Save screenshots (logging mode):
    if lgcLogMde:

        print('Saving screenshots')

        # Check whether target directory for frames (screenshots) for frames
        # exists, if not create it:
        lgcDir = os.path.isdir(strPthFrm)

        # If directory does not exist, create it:
        if not(lgcDir):
            # Create direcotry for segments:
            os.mkdir(strPthFrm)

        print(strPthFrm)

        # Save stimulus frame array to npy file:
        aryFrames = aryFrames.astype(np.int16)
        np.savez_compressed((strPthFrm
                             + os.path.sep
                             + 'stimFramesRun'),
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
                         + '_frame_'
                         + str(idxVol + 1).zfill(3)
                         + '.png')

            # Save image to disk:
            im.save(strTmpPth)

    # *************************************************************************
    # *** End of the experiment

    # Make the mouse cursor visible again:
    event.Mouse(visible=True)

    # Close everyting:
    objWin.close()
    core.quit()
    monitors.quit()
    logging.quit()
    event.quit()

# *****************************************************************************
# *** Function definitions


def func_exit():
    """
    Check whether exit-keys have been pressed.

    The exit keys are 'e' and 'x'; they have to be pressed at the same time.
    This is supposed to make it less likely that the experiment is aborted
    by accident.
    """
    # Check keyboard, save output to temporary string:
    lstExit = event.getKeys(keyList=['e', 'x'], timeStamped=False)

    # Whether the list has the correct length (if nothing has happened lstExit
    # will have length zero):
    if len(lstExit) != 0:

        if ('e' in lstExit) and ('x' in lstExit):

            # Log end of experiment:
            logging.data('------Experiment aborted by user.------')

            # Make the mouse cursor visible again:
            event.Mouse(visible=True)

            # Close everyting:
            # objWin.close()
            core.quit()
            monitors.quit()
            logging.quit()
            event.quit()

            return 1

        else:
            return 0

    else:
        return 0


# *****************************************************************************

if __name__ == "__main__":

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
                'Spatial frequency [cyc per bar]': 1.5,
                'Distance between observer and monitor [cm]': 99.0,
                'Width of monitor [cm]': 30.0,
                'Width of monitor [pixels]': 1920,
                'Height of monitor [pixels]': 1200,
                'Background colour [-1 to 1]': 0.0,
                'Show fixation grid?': [False, True]}

    if not(strFleNme is None):

        # If an input file name is provided, put it into the dictionary (as
        # default, can still be overwritten by user).
        dicParam['Run (name of design matrix file)'] = strFleNme

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
                                 (dicParam['Run (name of design matrix file)']
                                  + '_'
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

        # Add path for stimulus log (screenshots) for analysis to dictionary:
        strPthFrm = os.path.join(strPth,
                                 'log',
                                 (dicParam['Run (name of design matrix file)']
                                  + '_frames')
                                 )
        dicParam['Output path stimulus log (frames)'] = strPthFrm

        prf_stim(dicParam)

    else:

        # Close GUI if user presses 'cancel':
        core.quit()
