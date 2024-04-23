from typing import Protocol

import numpy as np

from ...core.individual import Individual
from ...core.population import Population


class VariationalOperator(Protocol):
    def __call__(self, population: Population) -> Population:
        pass


class GaussianMutation(VariationalOperator):
    def __init__(self, std: float, bounds: np.ndarray, probability: float) -> None:
        self.stds = np.full(len(bounds), std)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]
        self.probability = probability

    def __call__(self, population: Population) -> Population:
        new_population = population.copy()
        noise = np.random.normal(0, self.stds, size=new_population.genomes.shape)
        binary_mask = np.random.rand(*new_population.genomes.shape) < self.probability
        new_genomes = new_population.genomes + binary_mask * noise
        new_genomes = np.clip(new_genomes, self.lower_bounds, self.upper_bounds)
        new_population.update_genome(new_genomes)
        new_population.evaluate()
        return new_population


class TournamentSelection(VariationalOperator):
    def __init__(self, k: int = 2) -> None:
        self.k = k

    def __call__(self, population: Population) -> Population:
        population_copy = population.copy()
        num_individuals = len(population_copy.fitnesses)
        tournament_indices = np.random.randint(0, num_individuals, (num_individuals, self.k))
        tournament_fitnesses = population_copy.fitnesses[tournament_indices]
        selected_indices = (
            np.argmax(tournament_fitnesses, axis=1)
            if population_copy.problem.maximize
            else np.argmin(tournament_fitnesses, axis=1)
        )
        winners = tournament_indices[np.arange(num_individuals), selected_indices]
        new_genomes = population_copy.genomes[winners]
        return Population(new_genomes, population_copy.fitnesses[winners], population_copy.problem)


class SEA:
    def __init__(
        self,
        variational_operators_pipeline: list[VariationalOperator],
        k: int = 1,
    ) -> None:
        self.variational_operators_pipeline = variational_operators_pipeline
        self.k = k

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        offspring_population = parent_population.copy()
        for variational_operator in self.variational_operators_pipeline:
            offspring_population = variational_operator(offspring_population)
        topk_parent_population = parent_population.topk(self.k)
        total_offspring_population = offspring_population.merge(topk_parent_population)
        return total_offspring_population.topk(parent_population.size).to_individuals()

    @classmethod
    def create(self, **kwargs) -> "SEA":
        problem = kwargs.get("problem")
        mutation_std = kwargs.get("mutation_std")
        return SEA(
            variational_operators_pipeline=[
                TournamentSelection(),
                GaussianMutation(std=mutation_std, bounds=problem.bounds, probability=1.0),
            ],
            k=1,
        )
