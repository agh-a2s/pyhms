from leap_ec import Individual
from pyhms.config import BaseLevelConfig, CMALevelConfig, DELevelConfig, EALevelConfig, LocalOptimizationConfig
from structlog.typing import FilteringBoundLogger

from .abstract_deme import AbstractDeme
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .local_deme import LocalDeme


def init_root(config: BaseLevelConfig, logger: FilteringBoundLogger) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None, logger)


def init_from_config(
    config: BaseLevelConfig,
    new_id: str,
    target_level: int,
    metaepoch_count: int,
    seed: Individual,
    logger: FilteringBoundLogger,
) -> AbstractDeme:
    args = {
        "id": new_id,
        "level": target_level,
        "config": config,
        "started_at": metaepoch_count,
        "seed": seed,
        "logger": logger,
    }
    if isinstance(config, DELevelConfig):
        child = DEDeme(**args)
    elif isinstance(config, EALevelConfig):
        child = EADeme(**args)
    elif isinstance(config, CMALevelConfig):
        args["x0"] = seed
        args.pop("seed", None)
        child = CMADeme(**args)
    elif isinstance(config, LocalOptimizationConfig):
        child = LocalDeme(**args)
    return child
