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


def main():
    """py_pRF_mapping entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace:
    objParser.add_argument('-yeah', metavar='yolo', default='tutu',
                           help='Have a nice day.')

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    print('type(objNspc)')
    print(type(objNspc))

    print('objNspc')
    print(objNspc)

    # objParser.add_argument(
    #     'config', metavar='analysis/config.py',
    #     help=('Path to config file (if not using config file in standard \
    #            location, i.e. ~/analysis/config.py.')
    #            )

    # print(lstArgs)

    # lstArgs = objParser.parse_lstArgs()

    # print(lstArgs)

#...

    # Close file:
    # fleConfig.close()

if __name__ == "__main__":
    main()
