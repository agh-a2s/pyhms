from typing import List

from .gsc import root_stopped
from .single_pop.null_ea import NullEA
from .usc import dont_run, metaepoch_limit
from .sprout import far_enough, level_limit
from .config import AbstractLevelConfig, EALevelConfig, TreeConfig
from .tree import DemeTree

def hms(level_config: List[AbstractLevelConfig], gsc=metaepoch_limit(10),
        sprout_cond=far_enough(1.0)):
    config = TreeConfig(level_config, gsc, sprout_cond)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree

def local_optimization(x0, problem, bounds):
    level_config = [
        EALevelConfig(
            ea_class=NullEA, 
            seed=x0,
            pop_size=1, 
            problem=problem, 
            bounds=bounds, 
            lsc=dont_run(), 
            run_minimize=True
            )
        ]
    gsc = root_stopped()
    sprout_condition = level_limit(0)
    return hms(level_config, gsc, sprout_condition)
