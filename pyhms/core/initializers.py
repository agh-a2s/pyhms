from abc import ABC, abstractmethod

import numpy as np
from pyhms.core.individual import Individual
from pyhms.core.problem import Problem
from pyhms.utils.samplers import sample_normal, sample_uniform
from scipy.stats.qmc import LatinHypercube, Sobol


def linear_scale(sample: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + sample * (bounds[:, 1] - bounds[:, 0])


class PopInitializer(ABC):
    def __init__(self, bounds: np.ndarray | None = None):
        self._bounds = bounds

    @abstractmethod
    def sample_pop(self, pop_size: int | None) -> np.ndarray:
        pass

    def __call__(self, pop_size: int | None, problem: Problem):
        return [Individual(genome=genome, problem=problem) for genome in self.sample_pop(pop_size)]


class SeededPopInitializer(PopInitializer):
    @abstractmethod
    def get_seed(self, problem: Problem) -> Individual:
        pass


class UniformGlobalInitializer(PopInitializer):
    def __init__(self, bounds: np.ndarray):
        super().__init__(bounds)
        self.sampler = sample_uniform(bounds)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        return self.sampler(pop_size)


class LHSGlobalInitializer(PopInitializer):
    def __init__(self, bounds: np.ndarray, random_seed: int = None):
        super().__init__(bounds)
        self.sampler = LatinHypercube(d=len(bounds), seed=random_seed)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        sample = self.sampler.random(pop_size)
        return linear_scale(sample, self._bounds)


class SobolGlobalInitializer(PopInitializer):
    def __init__(self, bounds: np.ndarray, random_seed: int = None):
        super().__init__(bounds)
        self.sampler = Sobol(d=len(bounds), scramble=True, seed=random_seed)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        sample = self.sampler.random(pop_size)
        return linear_scale(sample, self._bounds)


class GaussianInitializer(PopInitializer):
    def __init__(self, seed: np.ndarray, std_dev: float, bounds: np.ndarray | None = None):
        super().__init__(bounds)
        self.sampler = sample_normal(seed, std_dev, bounds)
        self._seed = seed

    def sample_pop(self, pop_size: int) -> np.ndarray:
        return self.sampler(pop_size)


class GaussianInitializerWithSeedInject(SeededPopInitializer):
    def __init__(
        self, seed: Individual, std_dev: float, bounds: np.ndarray | None = None, preserve_fitness: bool = True
    ):
        super().__init__(bounds)
        self.sampler = sample_normal(seed.genome, std_dev, bounds)
        self._preserve_fitness = preserve_fitness
        self._seed_ind = seed

    def get_seed(self, problem: Problem) -> Individual:
        if self._preserve_fitness:
            return Individual(genome=self._seed_ind.genome, problem=problem, fitness=self._seed_ind.fitness)
        else:
            return Individual(genome=self._seed_ind.genome, problem=problem)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        return self.sampler(pop_size - 1)

    def __call__(self, pop_size: int | None, problem: Problem):
        return super().__call__(pop_size, problem) + [self.get_seed(problem)]


class InjectionInitializer(SeededPopInitializer):
    def __init__(
        self, injected_population: list[Individual], bounds: np.ndarray | None = None, preserve_fitness: bool = True
    ):
        super().__init__(bounds)
        self._preserve_fitness = preserve_fitness
        self.injected_population = injected_population

    def sample_pop(self, _: int | None) -> np.ndarray:
        raise NotImplementedError("InjectionInitializer does not have a sampler. Use __call__ instead.")

    def __call__(self, pop_size: int | None, problem: Problem) -> list[Individual]:
        if self._preserve_fitness:
            to_inject = [
                Individual(genome=ind.genome, problem=problem, fitness=ind.fitness) for ind in self.injected_population
            ]
        else:
            to_inject = [Individual(genome=ind.genome, problem=problem) for ind in self.injected_population]

        if pop_size is None or pop_size >= len(self.injected_population):
            return to_inject
        else:
            if self._preserve_fitness:
                return sorted(to_inject, reverse=True)[:pop_size]
            else:
                to_inject_array = np.array(to_inject)
                return to_inject_array[np.random.choice(len(to_inject), pop_size, replace=False)].tolist()

    def get_seed(self, problem: Problem) -> Individual:
        return self.__call__(1, problem)[0]
