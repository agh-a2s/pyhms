from .deme import Deme
from .tree import DemeTree

def level_limit(limit: int):
    def level_limit_sc(deme: Deme, level: int, tree: DemeTree):
        return len(tree.level(level)) < limit

    return level_limit_sc