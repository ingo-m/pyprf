# -*- coding: utf-8 -*-
"""Memory monitoring utility."""

# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
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

import sys
import numpy as np


def fncMemReport(varThr=10.0):
    """
    Print memory usage of variables in global namespace.

    Parameters
    ----------
    varThr : float
        Size threshold in MB; variables with a size below this are ignored.

    Returns
    -------
    This function has no return value.
    """
    # Get variables in namespace:
    dicMem = globals()

    # Dictionary for variable names & their size in MB:
    dicSze = {}

    # Loop through the dictionary returned by locals():
    for strTmp in dicMem.keys():

        # Get size of current variable in MB:
        varSze = np.around((sys.getsizeof(dicMem[strTmp]) * 0.000001),
                           decimals=3)

        # Put size of current variable into the size-dictionary:
        dicSze[strTmp] = varSze

    print('######################################')
    print('############ MEMORY USAGE ############')

    # Sort the size-dictionary:
    for strTmp in sorted(dicSze, key=dicSze.get, reverse=True):

        # Access size (in MB) of current element:
        varSze = dicSze[strTmp]

        # Print name of variable and its size if it is larger than threshold:
        if np.greater(varSze, varThr):

            strMsg = ('### Object: '
                      + strTmp
                      + ' --- Size: '
                      + str(varSze)
                      + ' MB')
            print(strMsg)

    print('######################################')
