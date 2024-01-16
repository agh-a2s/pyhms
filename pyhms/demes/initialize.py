from leap_ec import Individual
from pyhms.config import BaseLevelConfig, CMALevelConfig, DELevelConfig, EALevelConfig, LocalOptimizationConfig

from .abstract_deme import AbstractDeme
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .local_deme import LocalDeme


def init_root(config: BaseLevelConfig) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None)


def init_from_config(
    config: BaseLevelConfig, new_id: str, target_level: int, metaepoch_count: int, seed: Individual
) -> AbstractDeme:
    if isinstance(config, DELevelConfig):
        child = DEDeme(id=new_id, level=target_level, config=config, started_at=metaepoch_count, seed=seed)
    elif isinstance(config, EALevelConfig):
        child = EADeme(id=new_id, level=target_level, config=config, started_at=metaepoch_count, seed=seed)
    elif isinstance(config, CMALevelConfig):
        child = CMADeme(id=new_id, level=target_level, config=config, started_at=metaepoch_count, x0=seed)
    elif isinstance(config, LocalOptimizationConfig):
        child = LocalDeme(id=new_id, level=target_level, config=config, started_at=metaepoch_count, seed=seed)
    return child
