from typing import List

from .stop_conditions.gsc import root_stopped
from .demes.single_pop_eas.null_ea import NullEA
from .stop_conditions.usc import dont_run, metaepoch_limit
from .sprout import far_enough, level_limit
from .config import BaseLevelConfig, EALevelConfig, TreeConfig
from .tree import DemeTree

def hms(level_config: List[BaseLevelConfig], gsc=metaepoch_limit(10),
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
