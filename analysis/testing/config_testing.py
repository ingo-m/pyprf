"""Parameter definitions for testing."""

# Part of py_pRF_mapping library
# Copyright (C) 2017  Ingo Marquardt
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
varNumX = 10
# Number of y-positions to model:
varNumY = 10
# Number of pRF sizes to model:
varNumPrfSizes = 12

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
varPrfStdMin = 0.2
varPrfStdMax = 3.0

# Volume TR of input data [s]:
varTr = 2.832

# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.7

# Extent of temporal smoothing for fMRI data and pRF time course models
# [standard deviation of the Gaussian kernel, in seconds]:
varSdSmthTmp = 2.832

# Extent of spatial smoothing for fMRI data [standard deviation of the Gaussian
# kernel, in mm]
varSdSmthSpt = 0.5

# Perform linear trend removal on fMRI data?
lgcLinTrnd = True

# Number of fMRI volumes and png files to load:
varNumVol = 400

# Number of processes to run in parallel:
varPar = 4

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcSze = (200, 200)

# Path of functional data (needs to have same number of volumes as there are
# PNGs):
lstPathNiiFunc = ['/media/john/DATADRIVE1/MRI_Data_PhD/04_ParCon/20161212_02/nii_distcor/func_regAcrssRuns_cube/func_07.nii.gz']  #noqa

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/media/john/DATADRIVE1/MRI_Data_PhD/04_ParCon/20161212_02/nii_distcor/retinotopy/mask/func_07_mean_brainmask.nii.gz'  #noqa

# Output basename:
strPathOut = '/media/john/DATADRIVE1/MRI_Data_PhD/04_ParCon/20161212_02/nii_distcor/retinotopy/pRF_results/pRF_results'  #noqa

# Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding on
# CPU, 'gpu' for using GPU.
strVersion = 'numpy'

# Create pRF time course models?
lgcCrteMdl = True

# If we create new pRF time course models, the following parameters have to
# be provided:

# Basename of the 'binary stimulus files'. The files need to be in png
# format and number in the order of their presentation during the
# experiment.
strPathPng = '/media/john/DATADRIVE1/MRI_Data_PhD/04_ParCon/20161212_02/nii_distcor/retinotopy/pRF_stimuli/frame'  #noqa

# Start index of PNG files. For instance, `varStrtIdx = 0` if the name of
# the first PNG file is `file_000.png`, or `varStrtIdx = 1` if it is
# `file_001.png`.
varStrtIdx = 1

# Zero padding of PNG file names. For instance, `varStrtIdx = 3` if the
# name of PNG files is `file_007.png`, or `varStrtIdx = 4` if it is
# `file_0007.png`.
varZfill = 3

# Path to npy file with pRF time course models (to save or laod). Without file
# extension.
strPathMdl = '/media/john/DATADRIVE1/MRI_Data_PhD/04_ParCon/20161212_02/nii_distcor/retinotopy/pRF_results/pRF_model_tc'  #noqa
