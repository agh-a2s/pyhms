from pyhms.config import (
    BaseLevelConfig,
    CMALevelConfig,
    DELevelConfig,
    EALevelConfig,
    LocalOptimizationConfig,
    RandomLEvelConfig,
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
from .random_deme import RandomDeme


def init_root(config: BaseLevelConfig, logger: FilteringBoundLogger) -> AbstractDeme:
    return init_from_config(config, "root", 0, 0, None, None, logger)


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
    child_initializer: PopInitializer
    if config.pop_initializer_class == UniformGlobalInitializer:
        child_initializer = UniformGlobalInitializer(config.bounds)
    elif config.pop_initializer_class == GaussianInitializer:
        if hasattr(config, "sample_std_dev"):
            sample_std_dev = config.sample_std_dev
        else:
            sample_std_dev = get_default_mutation_std(config.bounds, target_level)
        child_initializer = GaussianInitializer(seed=sprout_seed.genome, std_dev=sample_std_dev, bounds=config.bounds)
    elif config.pop_initializer_class == GaussianInitializerWithSeedInject:
        if hasattr(config, "sample_std_dev"):
            sample_std_dev = config.sample_std_dev
        else:
            sample_std_dev = get_default_mutation_std(config.bounds, target_level)
        child_initializer = GaussianInitializerWithSeedInject(
            seed=sprout_seed, std_dev=sample_std_dev, bounds=config.bounds
        )
    elif config.pop_initializer_class == LHSGlobalInitializer:
        child_initializer = LHSGlobalInitializer(config.bounds, random_seed)
    elif config.pop_initializer_class == SobolGlobalInitializer:
        child_initializer = SobolGlobalInitializer(config.bounds, random_seed)
    elif config.pop_initializer_class == InjectionInitializer:
        if injected_population is None and sprout_seed is not None:
            injected_population = [sprout_seed]
        child_initializer = InjectionInitializer(injected_population, config.bounds)

    args = {
        "id": new_id,
        "level": target_level,
        "config": config,
        "initializer": child_initializer,
        "logger": logger,
        "started_at": metaepoch_count,
    }
    child: AbstractDeme
    if isinstance(config, DELevelConfig):
        child = DEDeme(**args)
    elif isinstance(config, EALevelConfig):
        child = EADeme(**args)
    elif isinstance(config, CMALevelConfig):
        args["random_seed"] = random_seed
        args["parent_deme"] = parent_deme
        child = CMADeme(**args)
    elif isinstance(config, LocalOptimizationConfig):
        child = LocalDeme(**args)
    elif isinstance(config, RandomLEvelConfig):
        child = RandomDeme(**args)
    return child
