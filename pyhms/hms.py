from typing import List

from .core.deme_config import BaseLevelConfig, TreeConfig
from .core.tree import DemeTree

def hms(level_config: List[BaseLevelConfig], gsc, sprout_cond, options={}):
    config = TreeConfig(level_config, gsc, sprout_cond, options=options)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree