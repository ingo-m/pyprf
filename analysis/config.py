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
varNumPrfSizes = 50

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 0.1
varPrfStdMax = 7.0

# Volume TR of input data [s]:
varTr = 2.832

# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.4

# Extent of temporal smoothing for fMRI data and pRF time course models
# [standard deviation of the Gaussian kernel, in seconds]:
varSdSmthTmp = 2.832

# Extent of spatial smoothing for fMRI data [standard deviation of the Gaussian
# kernel, in mm]
varSdSmthSpt = 0.0

# Perform linear trend removal on fMRI data?
lgcLinTrnd = True

# Number of fMRI volumes and png files to load:
varNumVol = 4 * 172

# Number of processes to run in parallel:
varPar = 11

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcSze = (300, 300)

# Path of functional data (needs to have same number of volumes as there are
# PNGs):
lstPathNiiFunc = ['/home/john/Desktop/tmp/func_regAcrssRuns_cube_up/func_07_up_aniso_smth.nii',
                  '/home/john/Desktop/tmp/func_regAcrssRuns_cube_up/func_08_up_aniso_smth.nii',
                  '/home/john/Desktop/tmp/func_regAcrssRuns_cube_up/func_09_up_aniso_smth.nii',
                  '/home/john/Desktop/tmp/func_regAcrssRuns_cube_up/func_10_up_aniso_smth.nii']  #noqa

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/mp2rage/04_seg/02_up/20161221_mp2rage_seg_v26.nii.gz'  #noqa

# Output basename:
strPathOut = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results_up/aniso_gamma_0p02_n_3/pRF_results'  #noqa

# List with paths of pickles with information about experimental design (order
# of stimuli). Only needed for pRF_motionLog.py (in order to create PNGs for
# static component of motion pRF mapping).
lstDsgn = ['~/ModBasedMotLoc/Conditions/Conditions_run01.pickle',
           '~/ModBasedMotLoc/Conditions/Conditions_run02.pickle',
           '~/ModBasedMotLoc/Conditions/Conditions_run03.pickle',
           '~/ModBasedMotLoc/Conditions/Conditions_run04.pickle']

# Path to npz file containing numpy array that defines stimulus shape, created
# with ~/py_pRF_motion/stimuli/Code/CreateMasks.py. Only needed for
# pRF_motionLog.py (in order to create PNGs for static component of motion pRF
# mapping).
strShpe = '~/mskBar.npz'

# Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding on
# CPU, 'gpu' for using GPU.
strVersion = 'cython'

# Create pRF time course models?
lgcCrteMdl = False

if lgcCrteMdl:
    # If we create new pRF time course models, the following parameters have to
    # be provided:

    # Basename of the 'binary stimulus files'. The files need to be in png
    # format and number in the order of their presentation during the
    # experiment.
    strPathPng = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_stimuli/frame_'  #noqa

    # Start index of PNG files. For instance, `varStrtIdx = 0` if the name of
    # the first PNG file is `file_000.png`, or `varStrtIdx = 1` if it is
    # `file_001.png`.
    varStrtIdx = 1

    # Zero padding of PNG file names. For instance, `varStrtIdx = 3` if the
    # name of PNG files is `file_007.png`, or `varStrtIdx = 4` if it is
    # `file_0007.png`.
    varZfill = 3

    # Output path for pRF time course models file (without file extension):
    strPathMdl = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results_up/pRF_model_tc'  #noqa

else:
    # If we use existing pRF time course models, the path to the respective
    # file has to be provided (including file extension, i.e. '*.npy'):
    strPathMdl = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20161221/nii_distcor/retinotopy/pRF_results_up/pRF_model_tc.npy'  #noqa
