import uuid
from copy import copy, deepcopy
from functools import total_ordering
from typing import Callable

import numpy as np

from .problem import Problem


@total_ordering
class Individual:
    def __init__(self, genome: np.ndarray, problem: Problem, fitness: float = np.nan):
        self.genome = genome
        self.fitness = fitness
        self.problem = problem
        self.uuid = uuid.uuid4()
        self.parents: set[uuid.UUID] = set()

    @classmethod
    def evaluate_population(cls, population: list["Individual"]) -> list["Individual"]:
        for individual in population:
            individual.evaluate()

        return population

    def evaluate(self) -> "Individual":
        if self.fitness is None or np.isnan(self.fitness):
            self.fitness = self.problem.evaluate(self.genome)
        return self

    def __lt__(self, other) -> bool:
        if other is None:
            return False
        return self.problem.worse_than(self.fitness, other.fitness)

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return self.problem.equivalent(self.fitness, other.fitness)

    @classmethod
    def create_population(cls, pop_size: int, problem: Problem, initialize: Callable) -> list["Individual"]:
        return [cls(genome=genome, problem=problem) for genome in initialize(pop_size)]

    def clone(self):
        cloned = copy(self)
        cloned.genome = deepcopy(self.genome)
        cloned.fitness = None
        cloned.parents = {self.uuid}
        return cloned
