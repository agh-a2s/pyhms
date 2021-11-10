import logging
import numpy as np
import numpy.random as nprand

from leap_ec import problem
from leap_ec.simple import ea_solve
from leap_ec.problem import FunctionProblem

from .gsc import metaepoch_limit
from .tree import DemeTree
from .config import LevelConfig
from .deme import Deme
from .single_pop.sea import SEA

def f(x):
    return sum(x**2)

logging.basicConfig(level=logging.DEBUG)

problem = FunctionProblem(f, maximize=False)
bounds = [(-10, 10) for _ in range(2)]

config = [
    LevelConfig(SEA(2, problem, bounds, pop_size=20)),
    LevelConfig(SEA(2, problem, bounds, pop_size=5, mutation_std=0.2), 0.1)
]
hms_tree = DemeTree(config, gsc=metaepoch_limit(10))
hms_tree.run()
for level, deme in hms_tree.all_demes:
    print(f"Level {level} {deme}")
    print(f"Best {max(deme.population)}")

#ea = sea(10, problem, bounds, pop_size=20)
#deme = Deme("root", LevelConfig(sea(10, problem, bounds, pop_size=20), problem, bounds))
#print(max(deme.population))
#for _ in range(10):
#    deme.run_metaepoch()
#    print(f"Best: {max(deme.population)}")
#    print(f"Centroid: {deme.centroid}")
#    print(f"Avg. fitness {deme.avg_fitness}")
