"""
    Global stopping conditions.
"""
from .tree import DemeTree

def dont_stop():
    return lambda tree: False
    