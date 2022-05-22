from typing import List
from abc import ABC

class AbstractLevelConfig(ABC):
    def __init__(self, problem, bounds, lsc):
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc

class EALevelConfig(AbstractLevelConfig):
    def __init__(self, ea_class, pop_size, problem, bounds, lsc, sample_std_dev=1.0, **kwargs) -> None:
        super().__init__(problem, bounds, lsc)
        self.ea_class = ea_class
        self.pop_size = pop_size
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)

class TreeConfig:
    def __init__(self, levels: List[AbstractLevelConfig], gsc, sprout_cond) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_cond = sprout_cond
