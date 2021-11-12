from typing import List

from .usc import metaepoch_limit
from .sprout import far_enough
from .config import LevelConfig
from .tree import DemeTree

def hms(level_config: List[LevelConfig], gsc=metaepoch_limit(10), 
    sprout_cond=far_enough(0.1)):

    hms_tree = DemeTree(level_config, gsc=gsc, sprout_cond=sprout_cond)
    hms_tree.run()
    local_optima = [leaf.best for leaf in hms_tree.leaves]

    return local_optima, hms_tree


