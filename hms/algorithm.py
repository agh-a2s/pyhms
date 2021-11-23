from typing import List

from .usc import metaepoch_limit
from .sprout import far_enough
from .config import LevelConfig, TreeConfig
from .tree import DemeTree

def hms(level_config: List[LevelConfig], gsc=metaepoch_limit(10), 
    sprout_cond=far_enough(1.0)):
    config = TreeConfig(level_config, gsc, sprout_cond)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree
