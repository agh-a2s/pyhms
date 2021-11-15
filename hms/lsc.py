"""
    Local stopping conditions.
"""
import numpy as np

from .deme import Deme

def fitness_steadiness(max_deviation:float=0.001, n_metaepochs:int=5):
    def stop_cond(deme: Deme) -> bool:
        if n_metaepochs > deme.metaepoch_count:
            return False
        
        avg_fits = [deme.avg_fitness(n) for n in range(-n_metaepochs, 0)]
        return max(abs(avg_fits - np.mean(avg_fits))) <= max_deviation

    return stop_cond

def all_children_stopped():
    def stop_cond(deme: Deme) -> bool:
        ch = deme.children
        return not (ch == []) and np.all([not c.active for c in ch])

    return stop_cond
