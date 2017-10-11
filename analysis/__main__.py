"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.py to set analysis parameters.
"""

import sys
import argparse


def main(args=None):
    """py_prf entry point."""
    if args is None:
        args = sys.argv[1:]
        print('no arguments')

    # Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config', metavar='analysis/config.py',
        help=('Path to config file (if not using config file in standard \
               location, i.e. ~/analysis/config.py.')
        )

if __name__ == "__main__":
    main()
