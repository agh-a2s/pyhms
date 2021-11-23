"""
    Universal stopping conditions. May be used as both local and global s.c.
"""
import toolz
from typing import Union

from .deme import Deme
from .tree import DemeTree

def metaepoch_limit_sc(obj: Union[Deme, DemeTree], limit: int) -> bool:
    return obj.metaepoch_count >= limit

def metaepoch_limit(limit: int):
    return toolz.curry(metaepoch_limit_sc, limit=limit)

def dont_stop_sc(_: Union[Deme, DemeTree]) -> bool:
    return False

def dont_stop():
    return dont_stop_sc
