"""
    Global stopping conditions.
"""
from typing import List, Union
import toolz

from .problem import StatsGatheringProblem
from .tree import DemeTree

def root_stopped_sc(tree: DemeTree) -> bool:
    return not tree.root.active

def root_stopped():
    return root_stopped_sc

def all_stopped_sc(tree: DemeTree) -> bool:
    return len(list(tree.active_demes)) == 0

def all_stopped():
    return all_stopped_sc

def fitness_eval_limit_reached_sc(
    tree: DemeTree, 
    limit: int, 
    weights: Union[List[float], str] = 'equal'
    ) -> bool:
    
    levels = tree.config.levels
    n_levels = len(levels)

    if weights == 'root':
        weights = [0 for _ in range(n_levels)]
        weights[0] = 1
    elif weights == 'equal' or weights is None:
        weights = [1 for _ in range(n_levels)]

    n_evals = 0    
    for i in range(n_levels):
        if not isinstance(levels[i].problem, StatsGatheringProblem):
            raise ValueError("Problem has to be an instance of StatsGatheringProblem")

        n_evals += weights[i] * levels[i].problem.n_evaluations

    return n_evals >= limit

def fitness_eval_limit_reached(limit: int, weights: List[float] = None):
    return toolz.curry(fitness_eval_limit_reached_sc, limit=limit, weights=weights)
