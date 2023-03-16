from typing import List

class BaseLevelConfig():
    def __init__(self, problem, lsc):
        self.problem = problem
        self.lsc = lsc

class EALevelConfig(BaseLevelConfig):
    def __init__(self, pop_size, problem, lsc, mutation_eta=20.0, sample_std_dev=1.0, **kwargs) -> None:
        super().__init__(problem, lsc)
        self.pop_size = pop_size
        self.mutation_eta = mutation_eta
        self.sample_std_dev = sample_std_dev
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)

class CMALevelConfig(BaseLevelConfig):
    def __init__(self, problem, lsc, sigma0, generations, **kwargs) -> None:
        super().__init__(problem, lsc)
        self.sigma0 = sigma0
        self.generations = generations
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.__dict__)

class TreeConfig:
    def __init__(self, levels: List[BaseLevelConfig], gsc, sprout_cond, **kwargs) -> None:
        self.levels = levels
        self.gsc = gsc
        self.sprout_cond = sprout_cond
        self.options = kwargs
