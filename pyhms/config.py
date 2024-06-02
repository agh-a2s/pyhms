from typing import Type, TypedDict

import numpy as np

from .core.initializers import InjectionInitializer, PopInitializer, SeededPopInitializer, UniformGlobalInitializer
from .core.problem import Problem
from .logging_ import LoggingLevel
from .stop_conditions import GlobalStopCondition, LocalStopCondition, UniversalStopCondition


class BaseLevelConfig:
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_initializer_type: Type[PopInitializer],
    ):
        self.problem = problem
        self.lsc = lsc
        self.pop_initializer_class = pop_initializer_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def bounds(self) -> np.ndarray:
        return self.problem.bounds


class EALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        ea_class,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        pop_initializer_type: Type[PopInitializer] = UniformGlobalInitializer,
        sample_std_dev: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc, pop_initializer_type)
        self.ea_class = ea_class
        self.pop_size = pop_size
        self.generations = generations
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)


class DELevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        pop_initializer_type: Type[PopInitializer] = UniformGlobalInitializer,
        sample_std_dev: float = 1.0,
        dither: bool = False,
        scaling: float = 0.8,
        crossover: float = 0.9,
    ):
        super().__init__(problem, lsc, pop_initializer_type)
        self.pop_size = pop_size
        self.generations = generations
        self.dither = dither
        self.scaling = scaling
        self.crossover = crossover
        self.sample_std_dev = sample_std_dev


class CMALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        sigma0: float | None,
        generations: int,
        pop_initializer_type: Type[SeededPopInitializer] = InjectionInitializer,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc, pop_initializer_type)
        self.sigma0 = sigma0
        self.generations = generations
        self.__dict__.update(kwargs)


class LocalOptimizationConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_initializer_type: Type[SeededPopInitializer] = InjectionInitializer,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc, pop_initializer_type)
        self.method = method
        self.__dict__.update(kwargs)


class RandomLevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_initializer_type: Type[PopInitializer],
        pop_size: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc, pop_initializer_type)
        self.pop_size = pop_size
        self.__dict__.update(kwargs)


class Options(TypedDict, total=False):
    log_level: LoggingLevel | None  # Default value: "warning"
    hibernation: bool | None  # Default value: False
    random_seed: int | None  # Default value: None


DEFAULT_OPTIONS: Options = {"log_level": LoggingLevel.WARNING, "hibernation": False}


class TreeConfig:
    def __init__(
        self,
        levels: list[BaseLevelConfig],
        gsc: GlobalStopCondition | UniversalStopCondition,
        sprout_mechanism,
        options: Options = DEFAULT_OPTIONS,
    ) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_mechanism = sprout_mechanism
        self.options = options
