from typing import List

import numpy as np
import numpy.typing as npt


class BaseLevelConfig:
    def __init__(self, problem, bounds: npt.NDArray[np.float64], lsc):
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc


class EALevelConfig(BaseLevelConfig):
    def __init__(self, ea_class, pop_size, problem, bounds, lsc, generations, sample_std_dev=1.0, **kwargs) -> None:
        super().__init__(problem, bounds, lsc)
        self.ea_class = ea_class
        self.pop_size = pop_size
        self.generations = generations
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)


class DELevelConfig(BaseLevelConfig):
    def __init__(self, pop_size, problem, bounds, lsc, generations, dither=False, scaling=0.8, crossover=0.9):
        super().__init__(problem, bounds, lsc)
        self.pop_size = pop_size
        self.generations = generations
        self.dither = dither
        self.scaling = scaling
        self.crossover = crossover


class CMALevelConfig(BaseLevelConfig):
    def __init__(self, problem, bounds, lsc, sigma0, generations, **kwargs) -> None:
        super().__init__(problem, bounds, lsc)
        self.sigma0 = sigma0
        self.generations = generations
        self.__dict__.update(kwargs)


class LocalOptimizationConfig(BaseLevelConfig):
    def __init__(self, problem, bounds, lsc, method="L-BFGS-B", **kwargs) -> None:
        super().__init__(problem, bounds, lsc)
        self.method = method
        self.__dict__.update(kwargs)


class TreeConfig:
    def __init__(self, levels: List[BaseLevelConfig], gsc, sprout_mechanism, options={}) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_mechanism = sprout_mechanism
        self.options = options
