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
from ..core.initializers import (
    GaussianInitializer,
    GaussianInitializerWithSeedInject,
    InjectionInitializer,
    LHSGlobalInitializer,
    PopInitializer,
    SobolGlobalInitializer,
    UniformGlobalInitializer,
)
from .abstract_deme import AbstractDeme
from .cma_deme import CMADeme
from .de_deme import DEDeme
from .ea_deme import EADeme
from .local_deme import LocalDeme
from .shade_deme import SHADEDeme


def init_root(config: BaseLevelConfig, logger: FilteringBoundLogger) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None, None, logger)


def create_initializer(
    config: BaseLevelConfig, sprout_seed: Individual | None, injected_population: list[Individual] | None
) -> PopInitializer:
    match config.pop_initializer_class.__qualname__:
        case UniformGlobalInitializer.__qualname__:
            return UniformGlobalInitializer(config.bounds)
        case GaussianInitializer.__qualname__:
            if hasattr(config, "sample_std_dev"):
                sample_std_dev = config.sample_std_dev
            else:
                sample_std_dev = get_default_mutation_std(config.bounds, 0)
            return GaussianInitializer(seed=sprout_seed.genome, std_dev=sample_std_dev, bounds=config.bounds)
        case GaussianInitializerWithSeedInject.__qualname__:
            if hasattr(config, "sample_std_dev"):
                sample_std_dev = config.sample_std_dev
            else:
                sample_std_dev = get_default_mutation_std(config.bounds, 0)
            return GaussianInitializerWithSeedInject(seed=sprout_seed, std_dev=sample_std_dev, bounds=config.bounds)
        case LHSGlobalInitializer.__qualname__:
            return LHSGlobalInitializer(config.bounds)
        case SobolGlobalInitializer.__qualname__:
            return SobolGlobalInitializer(config.bounds)
        case InjectionInitializer.__qualname__:
            if injected_population is None and sprout_seed is not None:
                injected_population = [sprout_seed]
            return InjectionInitializer(injected_population, config.bounds)
        case _:
            raise NotImplementedError(
                f"Creation of {config.pop_initializer_class.__qualname__} initializer not implemented"
            )


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
    child_initializer = create_initializer(config, sprout_seed, injected_population)

    args = {
        "id": new_id,
        "level": target_level,
        "config": config,
        "initializer": child_initializer,
        "logger": logger,
        "started_at": metaepoch_count,
    }

    match config:
        case DELevelConfig():
            return DEDeme(**args)
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
