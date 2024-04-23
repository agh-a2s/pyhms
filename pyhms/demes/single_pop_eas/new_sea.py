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
        # By default we use toroidal method, because it works the best for BBOB.
        new_genomes = self.apply_bounds(new_genomes, method="toroidal")
        new_population.update_genome(new_genomes)
        new_population.evaluate()
        return new_population

    def apply_bounds(self, genomes: np.ndarray, method: str = "clip"):
        if method == "clip":
            return np.clip(genomes, self.lower_bounds, self.upper_bounds)
        elif method == "reflect":
            broadcasted_lower_bounds = self.lower_bounds + np.zeros_like(genomes)
            broadcasted_upper_bounds = self.upper_bounds + np.zeros_like(genomes)
            over_upper = genomes > self.upper_bounds
            genomes[over_upper] = 2 * broadcasted_upper_bounds[over_upper] - genomes[over_upper]
            under_lower = genomes < self.lower_bounds
            genomes[under_lower] = 2 * broadcasted_lower_bounds[under_lower] - genomes[under_lower]
            return genomes
        elif method == "toroidal":
            range_size = self.upper_bounds - self.lower_bounds
            return self.lower_bounds + (genomes - self.lower_bounds) % range_size
        else:
            raise ValueError(f"Unknown method: {method}")


class UniformMutation(VariationalOperator):
    def __init__(self, bounds: np.ndarray, probability: float) -> None:
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]
        self.probability = probability

    def __call__(self, population: Population) -> Population:
        population_copy = population.copy()
        new_genomes = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(population.size, len(self.lower_bounds)),
        )
        new_genomes = np.where(
            np.random.rand(*new_genomes.shape) < self.probability,
            new_genomes,
            population_copy.genomes,
        )
        population_copy.update_genome(new_genomes)
        population_copy.evaluate()
        return population_copy


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
