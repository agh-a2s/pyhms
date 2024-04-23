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
        noise = np.random.normal(0, self.stds, size=population.genomes.shape)
        binary_mask = np.random.rand(*population.genomes.shape) < self.probability
        new_genomes = population.genomes + binary_mask * noise
        new_genomes = np.clip(new_genomes, self.lower_bounds, self.upper_bounds)
        population.update_genome(new_genomes)
        population.evaluate()
        return population


class TournamentSelection(VariationalOperator):
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def __call__(self, population: Population) -> Population:
        num_individuals = len(population.fitnesses)
        tournament_indices = np.random.randint(0, num_individuals, (num_individuals, self.k))
        tournament_fitnesses = population.fitnesses[tournament_indices]
        selected_indices = (
            np.argmax(tournament_fitnesses, axis=1)
            if population.problem.maximize
            else np.argmin(tournament_fitnesses, axis=1)
        )
        winners = tournament_indices[np.arange(num_individuals), selected_indices]
        new_genomes = population.genomes[winners]
        return Population(new_genomes, population.fitnesses[winners], population.problem)


class SEA:
    def __init__(self, variational_operators_pipeline: list[VariationalOperator]) -> None:
        self.variational_operators_pipeline = variational_operators_pipeline

    def run(self, parents: list[Individual]) -> list[Individual]:
        population = Population.from_individuals(parents)
        for variational_operator in self.variational_operators_pipeline:
            population = variational_operator(population)
        return population.to_individuals()
