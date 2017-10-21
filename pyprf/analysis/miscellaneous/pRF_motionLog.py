# -*- coding: utf-8 -*-
"""Module for pRF motion stimulus log."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
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

import numpy as np
import pickle
from PIL import Image
import config


class MotionLog:
    """
    Create stimulus log for motion pRF stimuli.

    The purpose of this module is to create the stimulus log for motion pRF
    stimuli, as used for example in the PacMan experiment. The information
    about stimulus shape and the information about the order of stimuli in the
    experiment are combined to create the complete stimulus log.

    For consistency with earlier versions, the numbering of frames (PNG files
    corresponding to fMRI volumes) starts at '1' (not at '0').
    """

    def __init__(self, strPthMsk):
        """Initialise instance of MotionLog class.

        Parameters
        ----------
        strPthMsk : strPthMsk
            Path to npz file containing numpy array that defines stimulus
            shape, created with `~/py_pRF_motion/stimuli/Code/CreateMasks.py`.
        """
        # Path of mask that outlines shape of visual stimulus:
        self.path_mask = strPthMsk

        # The mask itself:
        self.mask = None

        # First column of the design matrix (order of stimulus shapes):
        self.design_shp = []

        # Second column of the design matrix (order of motion directions):
        self.design_dir = []

    def loadMsk(self):
        """Load mask from npz file."""
        # Load npz file content into list:
        with np.load(self.path_mask) as objMsks:
            lstMsks = objMsks.items()

        for objTmp in lstMsks:
            strMsg = 'Mask type: ' + objTmp[0]
            # The following print statement prints the name of the mask stored
            # in the npz array from which the mask shape is retrieved. Can be
            # used to check whether the correct mask has been retrieved.
            print(strMsg)
            self.mask = objTmp[1].astype(np.int8)

    def addRun(self, strPckl, strKey='Conditions'):
        """Load design matrix (stimulus order) for one run from pickle.

        Parameters
        ----------
        strPckl : str
            Path of pickle containing dictionary with stimulus order. Stimulus
            order is defined in a two-dimensional numpy array, with rows
            corresponding to time [number of volumes] and the second dimension
            corresponding to two stimulus features: 1st, the stimulus type
            (in the same order as in the mask-array), 2nd the direction of
            movement (can be ignored for simple pRF analysis).
        strKey : str
            Dictionary key of numpy array with stimulus order.
        """
        # Load dictionary from pickle:
        with open(strPckl, 'r') as objPcl:
            dictPic = pickle.load(objPcl)

        # Access design matrix:
        aryDsgn = dictPic[strKey]

        # Loop through volumes and append design matrices:
        for idxVol in range(aryDsgn.shape[0]):

            # Append order of stimulus shapes to first column:
            self.design_shp.append(aryDsgn[idxVol, 0])

            # Append order of motion directions to second column:
            self.design_dir.append(aryDsgn[idxVol, 1])

    def getStimOrder(self):
        """Get stimulus order design matrix.

        Returns
        -------
        aryDsgn : np.array
            Design matrix of stimulus order. One-dimensional numpy with
            stimulus order for all runs that have been added so far. Indices of
            stimuli in this design matrix are with respect the same as the
            indices of stimulus shapes in the mask (from npz file).
        """
        aryDsgn = np.array(self.design_shp)
        aryDsgn = aryDsgn.astype(np.int8)
        return aryDsgn
    
    def getMask(self):
        """Get array with stimulus shapes.

        Returns
        -------
        mask : np.array
            Three-dimensional np.array, where first two dimension correspond
            to space (x, y) and third dimension to stimulus index.
        """
        return self.mask

# If this module is called directly, 
if __name__ == "__main__":

    # Instantiate motion log object:
    objMtnLog = MotionLog(config.strShpe)

    # Load stimulus shape:
    objMtnLog.loadMsk()

    # Add runs:
    for strTmp in config.lstDsgn:
        objMtnLog.addRun(strTmp)

    # Retrieve design matrix with order of stimuli:
    aryDsgn = objMtnLog.getStimOrder()

    # Retrieve stimulus shapes:
    aryMask = objMtnLog.getMask()

    # Number of volumes in design matrix:
    varNumVol = aryDsgn.shape[0]

    # When presenting the stimuli, psychopy presents the stimuli from the array
    # 'upside down', compared to a representation of the array where the first
    # row of the first column is in the upper left. In order to get the PNGs to
    # have the same orientations as the stimuli on the screen during the
    # experiment, we need to flip the array. (See
    # ~/py_pRF_motion/stimuli/Main/prfStim_Motion*.py for the experiment script
    # in which this 'flip' occurs.)
    aryMask = np.flipud(aryMask)

    # Maximum intensity of output PNG:
    varScle = 255

    # Rescale array:
    aryMask = np.multiply(aryMask, varScle)
    aryMask = aryMask.astype(np.uint8)

    # Loop through volumes and save PNGs:
    for idxVol in range(varNumVol):

        # Stimulus on current volume:
        varTmpStim = aryDsgn[idxVol]

        # Create image:
        im = Image.fromarray(aryMask[:, :, varTmpStim])

        # File name (with leading zeros, e.g. '*_004' or '*_042'). For
        # consistency with earlier versions, the numbering of frames (PNG files
        # corresponding to fMRI volumes) starts at '1' (not at '0').
        strTmpPth = (config.strPathPng
                     + str(idxVol + 1).zfill(3) 
                     + '.png')

        # Save image to disk:
        im.save(strTmpPth)

