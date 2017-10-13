"""Dummy main.py for config import test."""

from load_config import load_config


class cls_set_config(object):
    """
    Set config parameters from dictionary into local namespace.

    Parameters
    ----------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.
    """

    def __init__(self, dicCnfg):
        """Set config parameters from dictionary into local namespace."""
        self.__dict__.update(dicCnfg)


# External input parameter - path of config file:
dicCnfg = load_config('/home/john/Desktop/test/config.csv')

cfg = cls_set_config(dicCnfg)

print(' ')

print('type(cfg)')
print(type(cfg))

print(' ')

print('cfg.varTr')
print(cfg.varTr)

print(' ')

print('cfg.lstPathNiiFunc')
print(cfg.lstPathNiiFunc)
