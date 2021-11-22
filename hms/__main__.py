import logging

from leap_ec import problem
from leap_ec.problem import FunctionProblem

from .gsc import all_stopped
from .persist.tree import DemeTreeData
from .lsc import all_children_stopped, fitness_steadiness
from .usc import metaepoch_limit
from .algorithm import hms
from .config import LevelConfig
from .single_pop.sea import SEA

def f(x):
    return sum(x**2)

logging.basicConfig(level=logging.DEBUG)

problem = FunctionProblem(f, maximize=False)
bounds = [(-10, 10) for _ in range(2)]

config = [
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=problem, 
        bounds=bounds, 
        pop_size=20, 
        lsc=all_children_stopped()
        ),
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=problem, 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=0.2, 
        sample_std_dev=0.1, 
        lsc=fitness_steadiness(max_deviation=0.1)
        )
]

tree = hms(level_config=config, gsc=all_stopped())

tree_data = DemeTreeData(tree)
tree_data.save_binary()

print("Local optima found:")
for o in tree.optima:
    print(o)

print("\nDeme info:")
for level, deme in tree.all_demes:
    print(f"Level {level} {deme}")
    print(f"Best {deme.best}")
    print(f"Average fitness in last population {deme.avg_fitness()}")
    print(f"Average fitness in first population {deme.avg_fitness(0)}")
