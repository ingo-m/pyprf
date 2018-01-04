"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.py to set analysis parameters.
"""

import os
import argparse
from pyprf.analysis.pyprf_main import pyprf
from pyprf import __version__


# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def main():
    """py_pRF_mapping entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'PyPRF ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - config file path:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # # Add argument to namespace - test flag:
    # objParser.add_argument('-test',
    #                        action='store_true',
    #                        help='Whether to run a test with pytest.'
    #                        )

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    # # Get test flag from argument parser ('True' if the '-test' flag is
    # # provided, otherwise 'False'):
    # lgcTest = objNspc.test

    # if lgcTest:

    #     print('Test mode initiated...')

    #     # Path of config file for tests:
    #     strCsvCnfg = (strDir + '/testing/config_testing.csv')

    #     # Signal test mode to lower functions:
    #     lgcTest = True

    # else:

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    # Print info if no config argument is provided.
    if strCsvCnfg is None:
        print('Please provide the file path to a config file, e.g.:')
        print('   pyprf -config /path/to/my_config_file.csv')

    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        # Call to main function, to invoke pRF analysis:
        pyprf(strCsvCnfg, lgcTest)


if __name__ == "__main__":
    main()
