import numpy as np
import scipy

from ...core.individual import Individual
from ...core.population import Population
from .common import VariationalOperator, apply_bounds


def get_randoms(population: Population) -> np.ndarray:
    choices = np.indices((population.size, population.size))[1]

    # Create a mask to exclude self from selections
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    # Apply mask to choices to exclude self-indices
    filtered_choices = choices[mask].reshape(population.size, population.size - 1)

    # For each individual, select 3 random indices (without replacing)
    indices = np.array([np.random.choice(row, size=3, replace=False) for row in filtered_choices])

    # Select the genomes based on these indices
    return population.genomes[indices]


class BinaryMutation(VariationalOperator):
    def __init__(self, f: float) -> None:
        self.f = f

    def __call__(self, population: Population) -> Population:
        randoms = get_randoms(population)
        donor = randoms[:, 0] + self.f * (randoms[:, 1] - randoms[:, 2])
        new_genomes = apply_bounds(donor, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)


class BinaryMutationWithDither(VariationalOperator):
    def __call__(self, population: Population) -> Population:
        randoms = get_randoms(population)
        scaling = np.random.uniform(0.5, 1, size=population.size)
        scaling = np.repeat(scaling[:, np.newaxis], len(population.problem.bounds), axis=1)
        donor = randoms[:, 0] + scaling * (randoms[:, 1] - randoms[:, 2])
        new_genomes = apply_bounds(donor, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)


class CurrentToPBestMutation:
    def __call__(
        self,
        population: Population,
        archive: Population,
        f: np.ndarray,
        p: np.ndarray,
    ) -> Population:
        if population.size < 4:
            return population
        sorted_fitness_indexes = (
            np.argsort(population.fitnesses)
            if not population.problem.maximize
            else np.argsort(-1 * population.fitnesses)
        )
        p_best = []
        for p_i in p:
            best_index = sorted_fitness_indexes[: max(2, int(round(p_i * population.size)))]
            p_best.append(np.random.choice(best_index))
        p_best_np = np.array(p_best)
        randoms = get_randoms(population)
        randoms_with_archive = get_randoms(population.merge(archive)) if archive is not None else randoms
        randoms_with_archive = randoms_with_archive[: len(randoms)]
        mutated_genomes = (
            population.genomes
            + f * (population.genomes[p_best_np] - population.genomes)
            + f * (randoms[:, 0] - randoms_with_archive[:, 1])
        )
        new_genomes = apply_bounds(mutated_genomes, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)


class Crossover:
    def __call__(
        self,
        population: Population,
        mutated_population: Population,
        probability: np.ndarray | float,
    ) -> Population:
        chosen = np.random.rand(*population.genomes.shape)
        j_rand = np.random.randint(0, population.genomes.shape[1])
        chosen[j_rand :: population.genomes.shape[1]] = 0  # noqa: E203
        new_genomes = np.where(chosen <= probability, mutated_population.genomes, population.genomes)
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        new_population = Population(new_genomes, new_fitness, population.problem)
        return new_population


class DE:
    def __init__(self, use_dither: bool, crossover_probability: float, f: float | None = None) -> None:
        self._crossover_probability = crossover_probability
        self._mutation = BinaryMutationWithDither() if use_dither else BinaryMutation(f=f)
        self._crossover = Crossover()

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        trial_population = parent_population.copy()
        trial_population = self._mutation(trial_population)
        trial_population = self._crossover(parent_population, trial_population, self._crossover_probability)
        trial_population.evaluate()
        new_population_indices = (
            (trial_population.fitnesses >= parent_population.fitnesses)
            if parent_population.problem.maximize
            else (trial_population.fitnesses <= parent_population.fitnesses)
        )
        return (
            trial_population[new_population_indices].merge(parent_population[~new_population_indices]).to_individuals()
        )


class SHADE:
    def __init__(self, memory_size: int, population_size: int):
        self._memory_size = memory_size
        self._m_cr = np.ones(memory_size) * 0.5
        self._m_f = np.ones(memory_size) * 0.5
        self._archive: Population | None = None
        self._all_indexes = list(range(memory_size))
        self._pop_size = population_size
        self._k = 0
        self._mutation = CurrentToPBestMutation()
        self._crossover = Crossover()

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        offspring_population = parent_population.copy()
        cr, f, p = self._get_params()
        offspring_population = self._mutation(offspring_population, self._archive, f.reshape(len(f), 1), p)
        offspring_population = self._crossover(parent_population, offspring_population, cr.reshape(len(f), 1))
        offspring_population.evaluate()
        new_population_indices = (
            (offspring_population.fitnesses >= parent_population.fitnesses)
            if parent_population.problem.maximize
            else (offspring_population.fitnesses <= parent_population.fitnesses)
        )
        new_population = offspring_population[new_population_indices].merge(parent_population[~new_population_indices])
        if self._archive is None:
            self._archive = parent_population[new_population_indices]
        else:
            self._archive = self._archive.merge(parent_population[new_population_indices])
        if new_population_indices.any():
            self._update_memory(
                cr,
                f,
                parent_population.fitnesses,
                offspring_population.fitnesses,
                new_population_indices,
            )
        return new_population.to_individuals()

    def _get_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.random.choice(self._all_indexes, self._pop_size)
        cr = np.random.normal(self._m_cr[r], 0.1, self._pop_size)
        cr = np.clip(cr, 0, 1)
        cr[self._m_cr[r] == 1] = 0
        f = scipy.stats.cauchy.rvs(loc=self._m_f[r], scale=0.1, size=self._pop_size)
        f[f > 1] = 0
        while sum(f <= 0) != 0:
            r = np.random.choice(self._all_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=self._m_f[r], scale=0.1, size=sum(f <= 0))
        p = np.random.uniform(low=2 / self._pop_size, high=0.2, size=self._pop_size)
        return cr, f, p

    def _update_memory(
        self,
        cr: np.ndarray,
        f: np.ndarray,
        fitness: np.ndarray,
        c_fitness: np.ndarray,
        indexes: np.ndarray,
    ):
        if self._archive.size > self._memory_size:
            random_indices = np.random.choice(np.arange(self._archive.size), self._memory_size, replace=False)
            self._archive = self._archive[random_indices]
        if max(cr) != 0:
            weights = np.abs(fitness[indexes] - c_fitness[indexes])
            weights /= np.sum(weights)
            self._m_cr[self._k] = np.sum(weights * cr[indexes])
        else:
            self._m_cr[self._k] = 1

        self._m_f[self._k] = np.sum(f[indexes] ** 2) / np.sum(f[indexes])

        self._k += 1
        if self._k == self._memory_size:
            self._k = 0
