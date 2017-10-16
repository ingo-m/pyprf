"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.py to set analysis parameters.
"""

import argparse
from pyprf_main import pyprf


def main():
    """py_pRF_mapping entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           default='./testing/config_testing.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis.'
                           )

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    print(strCsvCnfg)

    # Call to main function, to invoke pRF analysis:
    # py_prf(strCsvCnfg)


if __name__ == "__main__":
    main()
