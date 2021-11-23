"""
    Local stopping conditions.
"""
import numpy as np
import toolz

from .deme import Deme

def fitness_steadiness_sc(deme: Deme, max_deviation: float, n_metaepochs: int) -> bool:
    if n_metaepochs > deme.metaepoch_count:
        return False
    
    avg_fits = [deme.avg_fitness(n) for n in range(-n_metaepochs, 0)]
    return max(abs(avg_fits - np.mean(avg_fits))) <= max_deviation

def fitness_steadiness(max_deviation: float = 0.001, n_metaepochs: int = 5):
    return toolz.curry(
        fitness_steadiness_sc, 
        max_deviation=max_deviation, 
        n_metaepochs=n_metaepochs
        )

def all_children_stopped_sc(deme: Deme) -> bool:
    ch = deme.children
    return not (ch == []) and np.all([not c.active for c in ch])

def all_children_stopped():
    return all_children_stopped_sc
