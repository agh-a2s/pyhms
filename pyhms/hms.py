from typing import List

from .config import DEFAULT_OPTIONS, BaseLevelConfig, Options, TreeConfig
from .stop_conditions import GlobalStopCondition, UniversalStopCondition
from .tree import DemeTree


def hms(
    level_config: List[BaseLevelConfig],
    gsc: GlobalStopCondition | UniversalStopCondition,
    sprout_cond,
    options: Options = DEFAULT_OPTIONS,
):
    config = TreeConfig(level_config, gsc, sprout_cond, options=options)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree
