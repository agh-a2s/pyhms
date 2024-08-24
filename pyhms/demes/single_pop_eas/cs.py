import numpy as np

from ...core.individual import Individual
from ...core.population import Population
from .common import VariationalOperator, apply_bounds


class LevyFlight:
    def __init__(self, beta: float = 1.5):
        self.beta = beta

    def __call__(self, size: int) -> np.ndarray:
        sigma_u = (
            np.math.gamma(1 + self.beta)
            * np.sin(np.pi * self.beta / 2)
            / (
                np.math.gamma((1 + self.beta) / 2)
                * self.beta
                * 2 ** ((self.beta - 1) / 2)
            )
        ) ** (1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, sigma_v, size)
        step = u / np.abs(v) ** (1 / self.beta)
        return step


class CuckooSearch(VariationalOperator):
    def __init__(self, p: float = 0.25, alpha: float = 1.0, beta: float = 1.5):
        assert alpha >= 0, "alpha must be non-negative"
        assert 0 <= p <= 1, "pa must be in [0, 1]"
        assert beta > 0 and beta <= 2, "beta must be in (0, 2]"
        self.pa = p
        self.alpha = alpha
        self.levy_flight = LevyFlight(beta)

    def get_best_solution(self, population: Population) -> np.ndarray:
        best_index = (
            np.argmax(population.fitnesses)
            if population.problem.maximize
            else np.argmin(population.fitnesses)
        )
        return population.genomes[best_index]

    def abandon_nests(self, population: Population) -> Population:
        n_abandon = int(self.pa * population.size)
        abandon_indices = np.random.choice(population.size, n_abandon, replace=False)
        new_nests = np.random.uniform(
            population.problem.bounds[:, 0],
            population.problem.bounds[:, 1],
            size=(n_abandon, population.genomes.shape[1]),
        )
        population.genomes[abandon_indices] = new_nests
        population.fitnesses[abandon_indices] = np.nan
        return population

    def __call__(self, population: Population) -> Population:
        best_solution = self.get_best_solution(population)

        # Generate new solutions (but keep the current best)
        step_sizes = self.levy_flight(population.size)
        new_genomes = population.genomes + self.alpha * step_sizes[:, np.newaxis] * (
            population.genomes - best_solution
        )
        new_genomes = apply_bounds(new_genomes, population.problem.bounds, "reflect")

        # Evaluate new solutions
        new_population = Population(
            new_genomes, np.full(population.size, np.nan), population.problem
        )
        new_population.evaluate()

        # Replace some nests by constructing new solutions
        new_population = self.abandon_nests(new_population)
        new_population.evaluate()

        # Keep the best solutions
        combined_population = population.merge(new_population)
        sorted_indices = np.argsort(combined_population.fitnesses)
        if population.problem.maximize:
            sorted_indices = sorted_indices[::-1]
        best_indices = sorted_indices[: population.size]

        return combined_population[best_indices]


class CuckooSearchOptimizer:
    def __init__(self, pa: float = 0.25, alpha: float = 1.0, beta: float = 1.5):
        self.cuckoo_search = CuckooSearch(pa, alpha, beta)

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        new_population = self.cuckoo_search(parent_population)
        return new_population.to_individuals()
