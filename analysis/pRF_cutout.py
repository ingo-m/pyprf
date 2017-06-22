# -*- coding: utf-8 -*-
"""Cut out parts. Kept for reference."""


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


    # *************************************************************************
    # *** Resample pixel-time courses in high-res visual space
    # The Gaussian sampling of the pixel-time courses takes place in the
    # super-sampled visual space. Here we take the convolved pixel-time courses
    # into this space, for each time point (volume).

    print('------Resample pixel-time courses in high-res visual space')

    # Array for super-sampled pixel-time courses:
    aryPngDataHigh = np.zeros((cfg.tplVslSpcHighSze[0],
                               cfg.tplVslSpcHighSze[1],
                               cfg.varNumVol))

    # Loop through volumes:
    for idxVol in range(0, cfg.varNumVol):

        print(idxVol)

        # The following array describes the coordinates of the pixels in the
        # flattened array (i.e. "vecOrigPixVal"). In other words, these are the
        # row and column coordinates of the original pixel values.
        aryOrigPixCoo = np.zeros([int(tplPngSize[0] * tplPngSize[1]),
                                  2])

        # Range for the coordinates:
        vecRange = np.arange(0, tplPngSize[0])

        # X coordinates:
        vecCooX = np.repeat(vecRange, tplPngSize[0])

        # Y coordinates:
        vecCooY = np.tile(vecRange, tplPngSize[1])

        # Put the pixel coordinates into the respective array:
        aryOrigPixCoo[:, 0] = vecCooX
        aryOrigPixCoo[:, 1] = vecCooY

        # The following vector will contain the actual original pixel values:
        # vecOrigPixVal = np.zeros([1,
        #                           int(tplPngSize[0]
        #                               * tplPngSize[1])])



#        vecOrigPixVal = aryPixConv[:, :, idxVol]
        vecOrigPixVal = aryPngData[:, :, idxVol]



        vecOrigPixVal = vecOrigPixVal.flatten()

        # The sampling interval for the creation of the super-sampled pixel
        # data (complex numbers are used as a convention for inclusive
        # intervals in "np.mgrid()").:
        # varStpSzeX = (float(tplPngSize[0])
        #               / float(cfg.tplVslSpcHighSze[0]))
        # varStpSzeY = (float(tplPngSize[1])
        #               / float(cfg.tplVslSpcHighSze[1]))
        varStpSzeX = np.complex(cfg.tplVslSpcHighSze[0])
        varStpSzeY = np.complex(cfg.tplVslSpcHighSze[1])

        print('echo01')

        # The following grid has the coordinates of the points at which we
        # would like to re-sample the pixel data:
        aryPixGridX, aryPixGridY = np.mgrid[0:tplPngSize[0]:varStpSzeX,
                                            0:tplPngSize[1]:varStpSzeY]

        print('echo02')

        # The actual resampling:
        aryResampled = griddata(aryOrigPixCoo,
                                vecOrigPixVal,
                                (aryPixGridX, aryPixGridY),
                                method='nearest')

        # Put super-sampled pixel time courses into array:
        aryPngDataHigh[:, :, idxVol] = aryResampled
    # *************************************************************************
