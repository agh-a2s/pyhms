"""
    Global stopping conditions.
"""
from .tree import DemeTree

def metaepoch_limit(limit: int):
    def stop_cond(tree: DemeTree) -> bool:
        return tree.metaepoch_count >= limit

    return stop_cond
