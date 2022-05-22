import logging

from leap_ec.problem import FunctionProblem

from hms.gsc import no_active_nonroot_demes
from hms.problem import StatsGatheringProblem
from hms.problems.relay2d import relay
from hms.sprout import far_enough
from hms.config import EALevelConfig
from hms import hms
from hms.single_pop import SEA
from hms.usc import dont_stop
from hms.lsc import fitness_steadiness
from hms.persist import DemeTreeData

bounds = [(-100, 100), (0, 100)]

hms_config = [
    EALevelConfig(
        ea_class=SEA,
        generations=2, 
        problem=StatsGatheringProblem(FunctionProblem(relay, maximize=False)), 
        bounds=bounds, 
        pop_size=20,
        mutation_std=50.0,
        k_elites=0,
        lsc=dont_stop()
        ),
    EALevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=StatsGatheringProblem(FunctionProblem(relay, maximize=False)), 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=5.0, 
        sample_std_dev=5.0,
        lsc=fitness_steadiness(max_deviation=0.5),
        run_minimize=True
        )
]

gsc = no_active_nonroot_demes(5)

sprout_cond = far_enough(min_distance=5.0)

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=gsc, sprout_cond=sprout_cond)
    DemeTreeData(tree).save_binary("relay2d")

if __name__ == '__main__':
    main()
