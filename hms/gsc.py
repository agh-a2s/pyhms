"""
    Global stopping conditions.
"""
from .tree import DemeTree

def root_stopped_sc(tree: DemeTree) -> bool:
    return not tree.root.active

def root_stopped():
    return root_stopped_sc

def all_stopped_sc(tree: DemeTree) -> bool:
    return len(list(tree.active_demes)) == 0

def all_stopped():
    return all_stopped_sc
