import logging
import copy

from leap_ec import problem
from leap_ec.problem import FunctionProblem

from hms.sprout import far_enough

from .problem import StatsGatheringProblem, square
from .gsc import all_stopped, fitness_eval_limit_reached
from .persist.tree import DemeTreeData
from .lsc import all_children_stopped, fitness_steadiness
from .usc import dont_stop, metaepoch_limit
from .algorithm import hms
from .config import LevelConfig
from .single_pop.sea import SEA

logging.basicConfig(level=logging.DEBUG)

problem = StatsGatheringProblem(FunctionProblem(square, maximize=False))
bounds = [(-10, 10) for _ in range(2)]

# If one wants to count evaluations for different levels separately, one has to
# use different instances of problem at each level.
config = [
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=problem, 
        bounds=bounds, 
        pop_size=20, 
        lsc=dont_stop()
        ),
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=copy.deepcopy(problem), 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=0.2, 
        sample_std_dev=0.1, 
        lsc=fitness_steadiness(max_deviation=0.1)
        )
]

gsc=fitness_eval_limit_reached(limit=1000, weights=None)

sprout_cond=far_enough(0.1)

def main():
    tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

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

    print("\nEvaluation stats:")
    for i in range(len(tree.config.levels)):
        print(f"Level {i}")
        prb = tree.config.levels[i].problem
        print(f"Count: {prb.n_evaluations}")
        m, s = prb.duration_stats
        print(f"Time avg.: {m}")
        print(f"Time std. dev.: {s}")

if __name__ == "__main__":
    main()
