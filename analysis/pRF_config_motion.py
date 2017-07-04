"""Motion pRF mapping experiment parameter definitions."""

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

# Number of x-positions to model:
varNumX = 40
# Number of y-positions to model:
varNumY = 40
# Number of pRF sizes to model:
varNumPrfSizes = 40

# Extend of visual space from centre of the screen (i.e. from the fixation
# point) [degrees of visual angle]:
varExtXmin = -5.19
varExtXmax = 5.19
varExtYmin = -5.19
varExtYmax = 5.19

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 0.20
varPrfStdMax = 7.0

# Volume TR of input data [s]:
varTr = 2.832

# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.8

# Extent of temporal smoothing for fMRI data and pRF time course models
# [standard deviation of the Gaussian kernel, in seconds]:
varSdSmthTmp = 2.0

# Extent of spatial smoothing for fMRI data [standard deviation of the Gaussian
# kernel, in mm]
varSdSmthSpt = 0.4

# Number of fMRI volumes and png files to load:
varNumVol = 4 * 172

# Intensity cutoff value for fMRI time series. Voxels with a mean intensity
# lower than the value specified here are not included in the pRF model finding
# (this speeds up the calculation, and, more importatnly, avoids division by
# zero):
varIntCtf = 50.0

# Number of processes to run in parallel:
varPar = 10

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcHighSze = (300, 300)

# Path of functional data (needs to have same number of volumes as there are
# PNGs):
lstPathNiiFunc = ['/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/func_regAcrssRuns_cube/func_07.nii.gz',
                  '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/func_regAcrssRuns_cube/func_08.nii.gz',
                  '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/func_regAcrssRuns_cube/func_09.nii.gz',
                  '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/func_regAcrssRuns_cube/func_10.nii.gz']  #noqa

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/mask/brainmask.nii.gz'  #noqa

# Output basename:
strPathOut = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results/pRF_results'  #noqa

# List with paths of pickles with information about experimental design (order
# of stimuli). Only needed for pRF_motionLog.py (in order to create PNGs for
# static component of motion pRF mapping).
lstDsgn = ['/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/log/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions/Conditions_run01.pickle',
           '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/log/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions/Conditions_run02.pickle',
           '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/log/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions/Conditions_run03.pickle',
           '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/log/PacMan_Pilot3_20161220/ModBasedMotLoc/Conditions/Conditions_run04.pickle']

# Path to npz file containing numpy array that defines stimulus shape, created
# with ~/py_pRF_motion/stimuli/Code/CreateMasks.py. Only needed for
# pRF_motionLog.py (in order to create PNGs for static component of motion pRF
# mapping).
strShpe = '/home/john/PhD/GitHub/py_pRF_motion/stimuli/Masks/mskCircleBar.npz'

# Use cython (i.e. compiled code) for faster performance? (Requires cython to
# be installed.)
lgcCython = True

# Create pRF time course models?
lgcCrteMdl = True

if lgcCrteMdl:
    # If we create new pRF time course models, the following parameters have to
    # be provided:

    # Basename of the 'binary stimulus files'. The files need to be in png
    # format and number in the order of their presentation during the
    # experiment.
    strPathPng = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_stimuli/frame_'  #noqa

    # Output path for pRF time course models file (without file extension):
    strPathMdl = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results/pRF_model_tc'  #noqa

else:
    # If we use existing pRF time course models, the path to the respective
    # file has to be provided (including file extension, i.e. '*.npy'):
    strPathMdl = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results/pRF_model_tc.npy'  #noqa
