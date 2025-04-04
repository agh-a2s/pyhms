from typing import Type, TypedDict

import numpy as np

from .core.problem import Problem
from .demes.single_pop_eas.sea import SEA, BaseSEA
from .logging_ import LoggingLevel
from .stop_conditions import GlobalStopCondition, LocalStopCondition, UniversalStopCondition


class BaseLevelConfig:
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
    ):
        self.problem = problem
        self.lsc = lsc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def bounds(self) -> np.ndarray:
        return self.problem.bounds


class EALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        ea_class: Type[BaseSEA] = SEA,
        sample_std_dev: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
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
        sample_std_dev: float = 1.0,
        dither: bool = False,
        scaling: float = 0.8,
        crossover: float = 0.9,
    ):
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.generations = generations
        self.dither = dither
        self.scaling = scaling
        self.crossover = crossover
        self.sample_std_dev = sample_std_dev


class SHADELevelConfig(BaseLevelConfig):
    def __init__(
        self,
        pop_size: int,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        memory_size: int,
        sample_std_dev: float = 1.0,
    ):
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.generations = generations
        self.memory_size = memory_size
        self.sample_std_dev = sample_std_dev


class CMALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        sigma0: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.sigma0 = sigma0
        self.generations = generations
        self.__dict__.update(kwargs)


class LocalOptimizationConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.method = method
        self.__dict__.update(kwargs)


class LHSLevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_size: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.__dict__.update(kwargs)


class SobolLevelConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        lsc: LocalStopCondition | UniversalStopCondition,
        pop_size: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, lsc)
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
        config_class_to_deme_class: dict[type[BaseLevelConfig], "type[AbstractDeme]"] = {},  # type: ignore # noqa: F821
    ) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_mechanism = sprout_mechanism
        self.options = options
        self.config_class_to_deme_class = config_class_to_deme_class
