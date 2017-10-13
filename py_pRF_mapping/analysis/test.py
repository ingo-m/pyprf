"""Dummy main.py for config import test."""

from load_config import load_config
from utilities import cls_set_config


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
