"""
    Sprouting conditions.
"""
from typing import List, Union
import numpy.linalg as nla
import toolz

from .deme import Deme
from .tree import DemeTree

def level_limit_sprout_cond(deme: Deme, level: int, tree: DemeTree, limit: int) -> bool:
    return len(tree.level(level)) < limit

def level_limit(limit: int):
    return toolz.curry(level_limit_sprout_cond, limit=limit)

def far_enough_sprout_cond(deme: Deme, level: int, tree: DemeTree, 
    min_distance: Union[List[float], float], norm_ord: int) -> bool:

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

def far_enough(min_distance: Union[List[float], float], norm_ord: int = 2):
    return toolz.curry(
        far_enough_sprout_cond, 
        min_distance=min_distance, 
        norm_ord=norm_ord
        )
