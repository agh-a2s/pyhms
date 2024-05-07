import numpy as np

from ...core.individual import Individual
from ...core.population import Population
from .common import VariationalOperator, apply_bounds


class BinaryMutation(VariationalOperator):
    def __init__(self, f: float) -> None:
        self.f = f

    def __call__(self, population: Population) -> Population:
        randoms = population.genomes[np.random.randint(0, population.size, size=(population.size, 2))]
        donor = population.genomes + self.f * (randoms[:, 0] - randoms[:, 1])
        new_genomes = apply_bounds(donor, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)


class BinaryMutationWithDither(VariationalOperator):
    def __call__(self, population: Population) -> Population:
        randoms = population.genomes[np.random.randint(0, population.size, size=(population.size, 2))]
        scaling = np.random.uniform(0.5, 1, size=population.size)
        scaling = np.repeat(scaling[:, np.newaxis], len(population.problem.bounds), axis=1)
        donor = population.genomes + scaling * (randoms[:, 0] - randoms[:, 1])
        new_genomes = apply_bounds(donor, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)


class Crossover:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, population: Population, mutated_population: Population) -> Population:
        chosen = np.random.rand(*population.genomes.shape)
        j_rand = np.random.randint(0, population.size)
        chosen[j_rand :: population.size] = 0  # noqa: E203
        new_genomes = np.where(chosen <= self.probability, mutated_population.genomes, population.genomes)
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        new_population = Population(new_genomes, new_fitness, population.problem)
        return new_population


class DE:
    def __init__(self, use_dither: bool, crossover_probability: float, f: float | None = None) -> None:
        self._mutation = BinaryMutationWithDither() if use_dither else BinaryMutation(f=f)
        self._crossover = Crossover(crossover_probability)

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        offspring_population = parent_population.copy()
        offspring_population = self._mutation(offspring_population)
        offspring_population = self._crossover(parent_population, offspring_population)
        offspring_population.evaluate()
        total_population = parent_population.merge(offspring_population)
        return total_population.topk(parent_population.size).to_individuals()
