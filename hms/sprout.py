"""
    Sprouting conditions.
"""
from typing import List, Union
import numpy.linalg as nla

from .deme import Deme
from .tree import DemeTree

def level_limit(limit: int):
    def level_limit_sc(deme: Deme, level: int, tree: DemeTree):
        return len(tree.level(level)) < limit

    return level_limit_sc

def far_enough(min_distance: Union[List[float], float], norm_ord=2):
    def sprout_possible(deme: Deme, level: int, tree: DemeTree):
        child_siblings = tree.level(level + 1)
        child_seed = max(deme.population)
        if isinstance(min_distance, list):
            min_dist = min_distance[level]
        elif isinstance(min_distance, float):
            min_dist = min_distance
        else:
            raise ValueError("min_distance must be a float or a list of floats")
        for sibling in child_siblings:
            if nla.norm(child_seed.genome - sibling.centroid, ord=norm_ord) <= min_dist:
                return False
        return True

    return sprout_possible