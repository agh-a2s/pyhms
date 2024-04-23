from typing import Callable
import numpy as np
from .individual import Individual
from .problem import Problem


class Population:
    def __init__(self, genomes: np.ndarray, fitnesses: np.ndarray, problem: Problem):
        self.genomes = genomes
        self.fitnesses = fitnesses
        self.problem = problem

    def evaluate(self, *args, **kwargs) -> None:
        nan_mask = np.isnan(self.fitnesses)
        if np.any(nan_mask):
            for i, genome in enumerate(self.genomes[nan_mask]):
                self.fitnesses[nan_mask][i] = self.problem.evaluate(
                    genome, *args, **kwargs
                )

    def update_genome(self, new_genome: np.ndarray) -> None:
        change_mask = np.any(new_genome != self.genomes, axis=1)
        self.genomes[change_mask] = new_genome[change_mask]
        self.fitnesses[change_mask] = np.nan

    @classmethod
    def from_individuals(cls, individuals: list[Individual]) -> "Population":
        population = cls(
            np.array([ind.genome for ind in individuals]),
            np.array([ind.fitness for ind in individuals]),
            individuals[0].problem,
        )
        return population

    def to_individuals(self) -> list[Individual]:
        return [
            Individual(genome, fitness, self.problem)
            for genome, fitness in zip(self.genomes, self.fitnesses)
        ]
