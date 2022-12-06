from typing import List

from .config import BaseLevelConfig, TreeConfig
from .tree import DemeTree

def hms(level_config: List[BaseLevelConfig], gsc, sprout_cond):
    config = TreeConfig(level_config, gsc, sprout_cond)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree