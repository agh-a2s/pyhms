"""
    Global stopping conditions.
"""
from .tree import DemeTree

def dont_stop():
    return lambda tree: False
    
def root_stopped():
    def stop_cond(tree: DemeTree) -> bool:
        return not tree.root.active

    return stop_cond

def all_stopped():
    def stop_cond(tree: DemeTree) -> bool:
        return len(list(tree.active_demes)) == 0

    return stop_cond