"""
    Universal stopping conditions. May be used as both local and global s.c.
"""

from typing import Union

from .deme import Deme
from .tree import DemeTree

def metaepoch_limit(limit: int):
    def stop_cond(obj: Union[Deme, DemeTree]) -> bool:
        return obj.metaepoch_count >= limit

    return stop_cond
