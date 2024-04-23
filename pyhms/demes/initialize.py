from pyhms.config import (
    BaseLevelConfig,
    CMALevelConfig,
    DELevelConfig,
    EALevelConfig,
    LHSLevelConfig,
    LocalOptimizationConfig,
)
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual
from .abstract_deme import AbstractDeme
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .lhs_deme import LHSDeme
from .local_deme import LocalDeme


def init_root(config: BaseLevelConfig, logger: FilteringBoundLogger) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None, logger)


def init_from_config(
    config: BaseLevelConfig,
    new_id: str,
    target_level: int,
    metaepoch_count: int,
    sprout_seed: Individual,
    logger: FilteringBoundLogger,
    random_seed: int = None,
    parent_deme: AbstractDeme | None = None,
) -> AbstractDeme:
    args = {
        "id": new_id,
        "level": target_level,
        "config": config,
        "started_at": metaepoch_count,
        "sprout_seed": sprout_seed,
        "logger": logger,
    }
    child: AbstractDeme
    if isinstance(config, DELevelConfig):
        child = DEDeme(**args)
    elif isinstance(config, EALevelConfig):
        child = EADeme(**args)
    elif isinstance(config, CMALevelConfig):
        args["x0"] = sprout_seed
        args.pop("sprout_seed", None)
        args["random_seed"] = random_seed
        args["parent_deme"] = parent_deme
        child = CMADeme(**args)
    elif isinstance(config, LocalOptimizationConfig):
        child = LocalDeme(**args)
    elif isinstance(config, LHSLevelConfig):
        args["random_seed"] = random_seed
        child = LHSDeme(**args)
    return child
