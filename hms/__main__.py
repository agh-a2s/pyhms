import logging
import numpy as np
import numpy.random as nprand

from leap_ec import problem
from leap_ec.simple import ea_solve
from leap_ec.problem import FunctionProblem
from hms.algorithm import hms

from hms.sprout import far_enough

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

optima, tree = hms(level_config=config)

print("Local optima found:")
for o in optima:
    print(o)

print("\nDeme info:")
for level, deme in tree.all_demes:
    print(f"Level {level} {deme}")
    print(f"Best {deme.best}")
