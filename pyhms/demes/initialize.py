from pyhms.config import (
    BaseLevelConfig,
    CMALevelConfig,
    DELevelConfig,
    EALevelConfig,
    LHSLevelConfig,
    LocalOptimizationConfig,
    SHADELevelConfig,
    SobolLevelConfig,
)
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual
from .abstract_deme import AbstractDeme, DemeInitArgs
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .lhs_deme import LHSDeme
from .local_deme import LocalDeme
from .shade_deme import SHADEDeme
from .sobol_deme import SobolDeme

CONFIG_CLASS_TO_DEME_CLASS = {
    DELevelConfig: DEDeme,
    SHADELevelConfig: SHADEDeme,
    EALevelConfig: EADeme,
    CMALevelConfig: CMADeme,
    LocalOptimizationConfig: LocalDeme,
    LHSLevelConfig: LHSDeme,
    SobolLevelConfig: SobolDeme,
}


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
):
    deme_init_args = DemeInitArgs(
        id=new_id,
        level=target_level,
        config=config,
        started_at=metaepoch_count,
        sprout_seed=sprout_seed,
        logger=logger,
        random_seed=random_seed,
        parent_deme=parent_deme,
    )
    return CONFIG_CLASS_TO_DEME_CLASS[type(config)](deme_init_args)  # type: ignore[abstract]
