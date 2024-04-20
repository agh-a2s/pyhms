import random
from abc import ABC, abstractmethod
from math import isclose, isnan
from typing import Callable

import numpy as np


class Problem(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, genome: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def worse_than(self, first_fitness: float, second_fitness: float):
        raise NotImplementedError

    def equivalent(self, first_fitness: float, second_fitness: float) -> bool:
        if type(first_fitness) == float and type(second_fitness) == float:
            return isclose(first_fitness, second_fitness)
        else:
            return first_fitness == second_fitness


class FunctionProblem(Problem):
    def __init__(self, fitness_function: Callable, bounds: np.ndarray, maximize: bool) -> None:
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.maximize = maximize

    def evaluate(self, genome: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.fitness_function(genome, *args, **kwargs)

    def worse_than(self, first_fitness: float, second_fitness: float) -> bool:
        if isnan(first_fitness):
            if isnan(second_fitness):
                return random.choice([True, False])
            return True
        elif isnan(second_fitness):
            return False
        if self.maximize:
            return first_fitness < second_fitness
        else:
            return first_fitness > second_fitness
