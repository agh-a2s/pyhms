import logging

from .config_solver import erikkson, bounds
from ...config import LevelConfig
from ...algorithm import hms
from ...single_pop.sea import SEA
from ...usc import metaepoch_limit, dont_stop
from ...persist.tree import DemeTreeData

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

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=gsc)
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()
