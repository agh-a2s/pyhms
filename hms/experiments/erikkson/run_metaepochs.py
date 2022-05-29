import logging

from hms.stop_conditions.gsc import no_active_nonroot_demes
from hms.sprout import far_enough
from hms.experiments.erikkson.config_solver import erikkson, bounds
from hms.config import EALevelConfig
from hms import hms
from hms.single_pop import SEA
from hms.stop_conditions.usc import dont_stop
from hms.stop_conditions.lsc import fitness_steadiness
from hms.persist import DemeTreeData

hms_config = [
    EALevelConfig(
        ea_class=SEA,
        generations=2, 
        problem=erikkson(0), 
        bounds=bounds, 
        pop_size=20,
        mutation_std=5.0,
        k_elites=0,
        lsc=dont_stop()
        ),
    EALevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=erikkson(1), 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=0.5, 
        sample_std_dev=0.5, 
        lsc=fitness_steadiness(max_deviation=0.1),
        run_minimize=True
        )
]

gsc = no_active_nonroot_demes(5)

sprout_cond = far_enough(min_distance=0.5)

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=gsc, sprout_cond=sprout_cond)
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()
