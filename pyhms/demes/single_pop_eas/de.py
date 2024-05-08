import numpy as np
import scipy

from ...core.individual import Individual
from ...core.population import Population
from .common import VariationalOperator, apply_bounds


def get_randoms(population: Population) -> np.ndarray:
    all_indices = np.arange(population.size)
    indices = np.array([np.random.choice(all_indices, size=3, replace=False) for _ in range(population.size)])
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


class CurrentToPBestMutation(VariationalOperator):
    def __init__(self, f: np.ndarray | None, p: np.ndarray):
        self.f = f
        self.p = p

    def __call__(self, population: Population) -> Population:
        assert self.f is not None
        if population.size < 4:
            return population
        sorted_fitness_indexes = (
            np.argsort(population.fitnesses)
            if not population.problem.maximize
            else np.argsort(-1 * population.fitnesses)
        )
        p_best = []
        for p_i in self.p:
            best_index = sorted_fitness_indexes[: max(2, int(round(p_i * population.size)))]
            p_best.append(np.random.choice(best_index))
        p_best_np = np.array(p_best)
        randoms = get_randoms(population)
        mutated_genomes = (
            population.genomes
            + self.f * (population.genomes[p_best_np] - population.genomes)
            + self.f * (randoms[:, 0] - randoms[:, 1])
        )
        new_genomes = apply_bounds(mutated_genomes, population.problem.bounds, "reflect")
        new_fitness = np.where(
            np.all(new_genomes == population.genomes, axis=1),
            population.fitnesses,
            np.nan,
        )
        return Population(new_genomes, new_fitness, population.problem)

    def adapt(self, f: np.ndarray) -> None:
        self.f = f


class Crossover:
    def __init__(self, probability: np.ndarray | None) -> None:
        self.probability = probability

    def __call__(self, population: Population, mutated_population: Population) -> Population:
        assert self.probability is not None
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

    def adapt(self, probability: np.ndarray) -> None:
        self.probability = probability


class DE:
    def __init__(self, use_dither: bool, crossover_probability: float, f: float | None = None) -> None:
        self._mutation = BinaryMutationWithDither() if use_dither else BinaryMutation(f=f)
        self._crossover = Crossover(crossover_probability)

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        trial_population = parent_population.copy()
        trial_population = self._mutation(trial_population)
        trial_population = self._crossover(parent_population, trial_population)
        trial_population.evaluate()
        new_population_indices = (
            (trial_population.fitnesses >= parent_population.fitnesses)
            if parent_population.problem.maximize
            else (trial_population.fitnesses <= parent_population.fitnesses)
        )
        return (
            trial_population[new_population_indices].merge(parent_population[~new_population_indices]).to_individuals()
        )


class CustomDE:
    def __init__(self, dither: bool, scaling: float, crossover_prob: float):
        self._dither = dither
        self._scaling = scaling
        self._crossover_prob = crossover_prob

    def run(self, parents: list[Individual]) -> list[Individual]:
        problem = parents[0].problem
        bounds = problem.bounds
        donors = self._create_donor_vectors(np.array([ind.genome for ind in parents]), bounds)
        donors_pop = [Individual(donor, problem=problem) for donor in donors]
        Individual.evaluate_population(donors_pop)
        offspring = [self._crossover(parent, donor) for parent, donor in zip(parents, donors_pop)]
        return offspring

    def _create_donor_vectors(self, parents: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        randoms = parents[np.random.randint(0, len(parents), size=(len(parents), 2))]
        if self._dither:
            scaling = np.random.uniform(0.5, 1, size=len(parents))
            scaling = np.repeat(scaling[:, np.newaxis], len(bounds), axis=1)
            donor = parents + scaling * (randoms[:, 0] - randoms[:, 1])
        else:
            donor = parents + self._scaling * (randoms[:, 0] - randoms[:, 1])

        return apply_bounds(donor, bounds, "reflect")

    def _crossover(self, parent: Individual, donor: Individual) -> Individual:
        if parent > donor:
            return parent
        else:
            genome = np.array(
                [p if np.random.uniform() < self._crossover_prob else d for p, d in zip(parent.genome, donor.genome)]
            )
            offspring = Individual(genome, problem=parent.problem)
            offspring.evaluate()
            return offspring


class SHADE:
    def __init__(self, memory_size: int, population_size: int):
        self._memory_size = memory_size
        self._m_cr = np.ones(memory_size) * 0.5
        self._m_f = np.ones(memory_size) * 0.5
        self._archive = []
        self._all_indexes = list(range(memory_size))
        self._pop_size = population_size
        self._init_pop_size = population_size
        self._k = 0
        self._p = np.ones(population_size) * 0.11
        self._mutation = CurrentToPBestMutation(f=None, p=self._p)
        self._crossover = Crossover(None)

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        offspring_population = parent_population.copy()
        cr, f = self._get_params()
        self._mutation.adapt(f.reshape(len(f), 1))
        self._crossover.adapt(cr.reshape(len(f), 1))
        offspring_population = self._mutation(offspring_population)
        offspring_population = self._crossover(parent_population, offspring_population)
        offspring_population.evaluate()
        new_population_indices = (
            (offspring_population.fitnesses >= parent_population.fitnesses)
            if parent_population.problem.maximize
            else (offspring_population.fitnesses <= parent_population.fitnesses)
        )
        new_population = offspring_population[new_population_indices].merge(parent_population[~new_population_indices])
        if new_population_indices.any():
            self._update_memory(
                cr,
                f,
                parent_population.fitnesses,
                offspring_population.fitnesses,
                new_population_indices,
            )
        self._archive.extend(new_population.genomes[new_population_indices])
        return new_population.to_individuals()

    def _get_params(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.choice(self._all_indexes, self._pop_size)
        cr = np.random.normal(self._m_cr[r], 0.1, self._pop_size)
        cr = np.clip(cr, 0, 1)
        cr[self._m_cr[r] == 1] = 0
        f = scipy.stats.cauchy.rvs(loc=self._m_f[r], scale=0.1, size=self._pop_size)
        f[f > 1] = 0
        while sum(f <= 0) != 0:
            r = np.random.choice(self._all_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=self._m_f[r], scale=0.1, size=sum(f <= 0))
        return cr, f

    def _update_memory(
        self,
        cr: np.ndarray,
        f: np.ndarray,
        fitness: np.ndarray,
        c_fitness: np.ndarray,
        indexes: np.ndarray,
    ):
        weights = np.abs(fitness[indexes] - c_fitness[indexes])
        weights /= np.sum(weights)
        self._m_cr[self._k] = np.sum(weights * cr[indexes] ** 2) / np.sum(weights * cr[indexes])
        if np.isnan(self._m_cr[self._k]):
            self._m_cr[self._k] = 1
        self._m_f[self._k] = np.sum(weights * f[indexes] ** 2) / np.sum(weights * f[indexes])
        self._k += 1
        if self._k == self._memory_size:
            self._k = 0
