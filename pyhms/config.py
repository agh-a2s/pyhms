from typing import TypedDict

import numpy as np
from leap_ec.problem import Problem

from .logging_ import LoggingLevel
from .stop_conditions import GlobalStopCondition, LocalStopCondition, UniversalStopCondition


class BaseLevelConfig:
    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
    ):
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"


class EALevelConfig(BaseLevelConfig):
    def __init__(
        self,
        ea_class,
        pop_size: int,
        problem: Problem,
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        sample_std_dev: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(problem, bounds, lsc)
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
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
        generations: int,
        sample_std_dev: float = 1.0,
        dither: bool = False,
        scaling: float = 0.8,
        crossover: float = 0.9,
    ):
        super().__init__(problem, bounds, lsc)
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
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
        sigma0: float | None,
        generations: int,
        **kwargs,
    ) -> None:
        super().__init__(problem, bounds, lsc)
        self.sigma0 = sigma0
        self.generations = generations
        self.__dict__.update(kwargs)


class LocalOptimizationConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> None:
        super().__init__(problem, bounds, lsc)
        self.method = method
        self.__dict__.update(kwargs)


class QuadraticSurrogateConfig(BaseLevelConfig):
    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        lsc: LocalStopCondition | UniversalStopCondition,
    ):
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc


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
