from pyhms.config import (
    BaseLevelConfig,
    CMALevelConfig,
    DELevelConfig,
    EALevelConfig,
    LocalOptimizationConfig,
    RandomLevelConfig,
    SHADELevelConfig,
)
from pyhms.utils.parameter_calculation import get_default_mutation_std
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual
from .abstract_deme import AbstractDeme
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .local_deme import LocalDeme
from .random_deme import RandomDeme
from .shade_deme import SHADEDeme


def init_root(config: BaseLevelConfig, logger: FilteringBoundLogger) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None, None, logger)


def prepare_initializer(
        config: BaseLevelConfig, target_level: int, sprout_seed: Individual | None, injected_population: list[Individual] | None
        ) -> None:
    context = {}
    if sprout_seed is not None:
        context["seed_genome"] = sprout_seed.genome
        context["seed_ind"] = sprout_seed
    if injected_population is not None:
        context["injected_pop"] = injected_population
    elif injected_population is None and sprout_seed is not None:
        context["injected_pop"] = [sprout_seed]
    if hasattr(config, "sample_std_dev"):
        context["std_dev"] = config.sample_std_dev
    else:
        context["std_dev"] = get_default_mutation_std(config.bounds, target_level)
    
    config.pop_initializer.prepare_sampler(context)


def init_from_config(
    config: BaseLevelConfig,
    new_id: str,
    target_level: int,
    metaepoch_count: int,
    sprout_seed: Individual | None,
    injected_population: list[Individual] | None,
    logger: FilteringBoundLogger,
    random_seed: int = None,
    parent_deme: AbstractDeme | None = None,
) -> AbstractDeme:

    prepare_initializer(config, target_level, sprout_seed, injected_population)

    args = {
        "id": new_id,
        "level": target_level,
        "config": config,
        "logger": logger,
        "started_at": metaepoch_count,
    }

    match config:
        case DELevelConfig():
            return DEDeme(**args)
        case SHADELevelConfig():
            return SHADEDeme(**args)
        case EALevelConfig():
            return EADeme(**args)
        case CMALevelConfig():
            args["random_seed"] = random_seed
            args["parent_deme"] = parent_deme
            return CMADeme(**args)
        case LocalOptimizationConfig():
            return LocalDeme(**args)
        case RandomLevelConfig():
            return RandomDeme(**args)
        case _:
            raise NotImplementedError(f"Creation of {config.__class__.__name__} deme not implemented")
