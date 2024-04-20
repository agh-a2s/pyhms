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

    def __lt__(self, other: "Individual") -> bool:
        return self.problem.worse_than(self.fitness, other.fitness)
