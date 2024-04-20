from typing import Callable

import numpy as np

from .problem import Problem


class Individual:
    def __init__(self, genome: np.ndarray, problem: Problem):
        self.genome = genome
        self.fitness = None
        self.problem = problem

    @classmethod
    def evaluate_population(cls, population: list["Individual"]) -> list["Individual"]:
        for individual in population:
            individual.evaluate()

        return population

    def evaluate(self) -> "Individual":
        self.fitness = self.problem.evaluate(self.genome)
        return self

    def __lt__(self, other) -> bool:
        return self.problem.worse_than(self.fitness, other.fitness)

    def __eq__(self, other) -> bool:
        return np.isclose(self.fitness, other.fitness)

    @classmethod
    def create_population(cls, pop_size: int, problem: Problem, initialize: Callable) -> list["Individual"]:
        return [cls(genome=initialize(), problem=problem) for _ in range(pop_size)]
