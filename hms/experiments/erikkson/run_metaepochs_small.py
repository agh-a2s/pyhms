import logging

from hms.sprout import far_enough
from hms.experiments.erikkson.config_solver import erikkson, bounds
from hms.config import LevelConfig
from hms import hms
from hms.single_pop import SEA
from hms.usc import metaepoch_limit, dont_stop
from hms.persist import DemeTreeData

hms_config = [
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=erikkson(0), 
        bounds=bounds, 
        pop_size=20,
        lsc=dont_stop()
        ),
    LevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=erikkson(1), 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=0.2, 
        sample_std_dev=0.2, 
        lsc=metaepoch_limit(2)
        )
]

gsc = metaepoch_limit(10)

sprout_cond = far_enough(min_distance=1.0)

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=gsc, sprout_cond=sprout_cond)
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()
