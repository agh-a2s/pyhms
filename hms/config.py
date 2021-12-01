from typing import List

class LevelConfig:
    def __init__(self, ea_class, pop_size, problem, bounds, lsc, sample_std_dev=1.0, **kwargs) -> None:
        self.ea_class = ea_class
        self.pop_size = pop_size
        self.problem = problem
        self.bounds = bounds
        self.lsc = lsc
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)

class TreeConfig:
    def __init__(self, levels: List[LevelConfig], gsc, sprout_cond) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_cond = sprout_cond
