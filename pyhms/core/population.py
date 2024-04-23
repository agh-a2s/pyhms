import numpy as np

from .individual import Individual
from .problem import Problem


class Population:
    def __init__(self, genomes: np.ndarray, fitnesses: np.ndarray, problem: Problem):
        self.genomes = genomes
        self.fitnesses = fitnesses
        self.problem = problem

    @property
    def size(self) -> int:
        return len(self.genomes)

    def evaluate(self, *args, **kwargs) -> None:
        nan_mask = np.isnan(self.fitnesses)
        if np.any(nan_mask):
            for i, genome in enumerate(self.genomes[nan_mask]):
                self.fitnesses[nan_mask][i] = self.problem.evaluate(genome, *args, **kwargs)

    def update_genome(self, new_genome: np.ndarray) -> None:
        change_mask = np.any(new_genome != self.genomes, axis=1)
        self.genomes[change_mask] = new_genome[change_mask]
        self.fitnesses[change_mask] = np.nan

    def copy(self) -> "Population":
        new_genomes = np.copy(self.genomes)
        new_fitnesses = np.copy(self.fitnesses)
        new_population = Population(new_genomes, new_fitnesses, self.problem)
        return new_population

    @classmethod
    def from_individuals(cls, individuals: list[Individual]) -> "Population":
        population = cls(
            np.array([ind.genome for ind in individuals], dtype=float),
            np.array([ind.fitness for ind in individuals], dtype=float),
            individuals[0].problem,
        )
        return population

    def to_individuals(self) -> list[Individual]:
        return [Individual(genome, self.problem, fitness) for genome, fitness in zip(self.genomes, self.fitnesses)]

    def topk(self, k: int) -> "Population":
        topk_indices = (
            np.argsort(self.fitnesses)[-k:] if self.problem.maximize else np.argsort(self.fitnesses)[:k][::-1]
        )
        return Population(self.genomes[topk_indices], self.fitnesses[topk_indices], self.problem)

    def merge(self, other: "Population") -> "Population":
        new_genomes = np.concatenate((self.genomes, other.genomes))
        new_fitnesses = np.concatenate((self.fitnesses, other.fitnesses))
        return Population(new_genomes, new_fitnesses, self.problem)
