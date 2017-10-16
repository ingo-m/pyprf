"""Load py_pRF_mapping config file."""

import os
import csv
import ast

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def load_config(strCsvCnfg, lgcTest=False):  #noqa
    """
    Load py_pRF_mapping config file.

    Parameters
    ----------
    strCsvCnfg : string
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of this function
        will be prepended to config file paths.

    Returns
    -------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.
    """
    # Print config parameters?
    lgcPrint = True

    # Dictionary with config information:
    dicCnfg = {}

    # Open file with parameter configuration:
    # fleConfig = open(strCsvCnfg, 'r')
    with open(strCsvCnfg, 'r') as fleConfig:

        # Read file  with ROI information:
        csvIn = csv.reader(fleConfig,
                           delimiter='\n',
                           skipinitialspace=True)

        # Loop through csv object to fill list with csv data:
        for lstTmp in csvIn:

            # Skip comments (i.e. lines starting with '#') and empty lines.
            # Note: Indexing the list (i.e. lstTmp[0][0]) does not work for
            # empty lines. However, if the first condition is no fullfilled
            # (i.e. line is empty and 'if lstTmp' evaluates to false), the
            # second logical test (after the 'and') is not actually carried
            # out.
            if lstTmp and not (lstTmp[0][0] == '#'):

                # Name of current parameter (e.g. 'varTr'):
                strParamKey = lstTmp[0].split(' = ')[0]
                # print(strParamKey)

                # Current parameter value (e.g. '2.94'):
                strParamVlu = lstTmp[0].split(' = ')[1]
                # print(strParamVlu)

                # Put paramter name (key) and value (item) into dictionary:
                dicCnfg[strParamKey] = strParamVlu

    # Number of x-positions to model:
    dicCnfg['varNumX'] = int(dicCnfg['varNumX'])
    if lgcPrint:
        print('---Number of x-positions to model: ' + str(dicCnfg['varNumX']))

    # Number of y-positions to model:
    dicCnfg['varNumY'] = int(dicCnfg['varNumY'])
    if lgcPrint:
        print('---Number of y-positions to model: ' + str(dicCnfg['varNumY']))

    # Number of pRF sizes to model:
    dicCnfg['varNumPrfSizes'] = int(dicCnfg['varNumPrfSizes'])
    if lgcPrint:
        print('---Number of pRF sizes to model: '
              + str(dicCnfg['varNumPrfSizes']))

    # Extent of visual space from centre of the screen in negative x-direction
    # (i.e. from the fixation point to the left end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtXmin'] = float(dicCnfg['varExtXmin'])
    if lgcPrint:
        print('---Extent of visual space in negative x-direction: '
              + str(dicCnfg['varExtXmin']))

    # Extent of visual space from centre of the screen in positive x-direction
    # (i.e. from the fixation point to the right end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtXmax'] = float(dicCnfg['varExtXmax'])
    if lgcPrint:
        print('---Extent of visual space in positive x-direction: '
              + str(dicCnfg['varExtXmax']))

    # Extent of visual space from centre of the screen in negative y-direction
    # (i.e. from the fixation point to the lower end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtYmin'] = float(dicCnfg['varExtYmin'])
    if lgcPrint:
        print('---Extent of visual space in negative y-direction: '
              + str(dicCnfg['varExtYmin']))

    # Extent of visual space from centre of the screen in positive y-direction
    # (i.e. from the fixation point to the upper end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtYmax'] = float(dicCnfg['varExtYmax'])
    if lgcPrint:
        print('---Extent of visual space in positive y-direction: '
              + str(dicCnfg['varExtYmax']))

    # Minimum pRF model size (standard deviation of 2D Gaussian) [degrees of
    # visual angle]:
    dicCnfg['varPrfStdMin'] = float(dicCnfg['varPrfStdMin'])
    if lgcPrint:
        print('---Minimum pRF model size: ' + str(dicCnfg['varPrfStdMin']))

    # Maximum pRF model size (standard deviation of 2D Gaussian) [degrees of
    # visual angle]:
    dicCnfg['varPrfStdMax'] = float(dicCnfg['varPrfStdMax'])
    if lgcPrint:
        print('---Maximum pRF model size: ' + str(dicCnfg['varPrfStdMax']))

    # Volume TR of input data [s]:
    dicCnfg['varTr'] = float(dicCnfg['varTr'])
    if lgcPrint:
        print('---Volume TR of input data [s]: ' + str(dicCnfg['varTr']))

    # Voxel resolution of fMRI data [mm]:
    dicCnfg['varVoxRes'] = float(dicCnfg['varVoxRes'])
    if lgcPrint:
        print('---Voxel resolution of fMRI data [mm]: '
              + str(dicCnfg['varVoxRes']))

    # Extent of temporal smoothing for fMRI data and pRF time course models
    # [standard deviation of the Gaussian kernel, in seconds]:
    dicCnfg['varSdSmthTmp'] = float(dicCnfg['varSdSmthTmp'])
    if lgcPrint:
        print('---Extent of temporal smoothing (Gaussian SD in [s]): '
              + str(dicCnfg['varSdSmthTmp']))

    # Extent of spatial smoothing for fMRI data [standard deviation of the
    # Gaussian kernel, in mm]
    dicCnfg['varSdSmthSpt'] = float(dicCnfg['varSdSmthSpt'])
    if lgcPrint:
        print('---Extent of spatial smoothing (Gaussian SD in [mm]): '
              + str(dicCnfg['varSdSmthSpt']))

    # Perform linear trend removal on fMRI data?
    dicCnfg['lgcLinTrnd'] = (dicCnfg['lgcLinTrnd'] == 'True')
    if lgcPrint:
        print('---Linear trend removal: ' + str(dicCnfg['lgcLinTrnd']))

    # Number of fMRI volumes and png files to load:
    dicCnfg['varNumVol'] = int(dicCnfg['varNumVol'])
    if lgcPrint:
        print('---Total number of fMRI volumes and png files: '
              + str(dicCnfg['varNumVol']))

    # Number of processes to run in parallel:
    dicCnfg['varPar'] = int(dicCnfg['varPar'])
    if lgcPrint:
        print('---Number of processes to run in parallel: '
              + str(dicCnfg['varPar']))

    # Size of high-resolution visual space model in which the pRF models are
    # created (x- and y-dimension).
    dicCnfg['tplVslSpcSze'] = tuple([int(dicCnfg['varVslSpcSzeX']),
                                     int(dicCnfg['varVslSpcSzeY'])])
    if lgcPrint:
        print('---Size of high-resolution visual space model (x & y): '
              + str(dicCnfg['tplVslSpcSze']))

    # Path(s) of functional data:
    dicCnfg['lstPathNiiFunc'] = ast.literal_eval(dicCnfg['lstPathNiiFunc'])
    if lgcPrint:
        print('---Path(s) of functional data:')
        for strTmp in dicCnfg['lstPathNiiFunc']:
            print('   ' + str(strTmp))

    # Path of mask (to restrict pRF model finding):
    dicCnfg['strPathNiiMask'] = ast.literal_eval(dicCnfg['strPathNiiMask'])
    if lgcPrint:
        print('---Path of mask (to restrict pRF model finding):')
        print('   ' + str(dicCnfg['strPathNiiMask']))

    # Output basename:
    dicCnfg['strPathOut'] = ast.literal_eval(dicCnfg['strPathOut'])
    if lgcPrint:
        print('---Output basename:')
        print('   ' + str(dicCnfg['strPathOut']))

    # Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding
    # on CPU, 'gpu' for using GPU.
    dicCnfg['strVersion'] = ast.literal_eval(dicCnfg['strVersion'])
    if lgcPrint:
        print('---Version (numpy, cython, or gpu): '
              + str(dicCnfg['strVersion']))

    # Create pRF time course models?
    dicCnfg['lgcCrteMdl'] = (dicCnfg['lgcCrteMdl'] == 'True')
    if lgcPrint:
        print('---Create pRF time course models: '
              + str(dicCnfg['lgcCrteMdl']))

    # Path to npy file with pRF time course models (to save or laod). Without
    # file extension.
    dicCnfg['strPathMdl'] = ast.literal_eval(dicCnfg['strPathMdl'])
    if lgcPrint:
        print('---Path to npy file with pRF time course models (to save '
              + 'or load):')
        print('   ' + str(dicCnfg['strPathMdl']))

    # If we create new pRF time course models, the following parameters have to
    # be provided:
    if dicCnfg['lgcCrteMdl']:

        # Basename of the 'binary stimulus files'. The files need to be in png
        # format and number in the order of their presentation during the
        # experiment.
        dicCnfg['strPathPng'] = ast.literal_eval(dicCnfg['strPathPng'])
        if lgcPrint:
            print('---Basename of PNG stimulus files: '
                  + str(dicCnfg['strPathPng']))

        # Start index of PNG files. For instance, `varStrtIdx = 0` if the name
        # of the first PNG file is `file_000.png`, or `varStrtIdx = 1` if it is
        # `file_001.png`.
        dicCnfg['varStrtIdx'] = int(dicCnfg['varStrtIdx'])
        if lgcPrint:
            print('---Start index of PNG files: '
                  + str(dicCnfg['varStrtIdx']))

        # Zero padding of PNG file names. For instance, `varStrtIdx = 3` if the
        # name of PNG files is `file_007.png`, or `varStrtIdx = 4` if it is
        # `file_0007.png`.
        dicCnfg['varZfill'] = int(dicCnfg['varZfill'])
        if lgcPrint:
            print('---Zero padding of PNG file names: '
                  + str(dicCnfg['varZfill']))

    # Is this a test?
    if lgcTest:

        # Prepend absolute path of this file to config file paths:
        dicCnfg['strPathNiiMask'] = (strDir + dicCnfg['strPathNiiMask'])
        dicCnfg['strPathOut'] = (strDir + dicCnfg['strPathOut'])
        dicCnfg['strPathPng'] = (strDir + dicCnfg['strPathPng'])
        dicCnfg['strPathMdl'] = (strDir + dicCnfg['strPathMdl'])

        # Loop through functional runs:
        varNumRun = len(dicCnfg['lstPathNiiFunc'])
        for idxRun in range(varNumRun):
            dicCnfg['lstPathNiiFunc'][idxRun] = (
                strDir
                + dicCnfg['lstPathNiiFunc'][idxRun]
                )

    return dicCnfg
