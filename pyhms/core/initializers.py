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
        self._sampler: callable = None

    @abstractmethod
    def prepare_sampler(self, context: dict = None) -> None:
        pass

    def sample_pop(self, pop_size: int | None) -> np.ndarray:
        return self._sampler(pop_size)

    def __call__(self, pop_size: int | None, problem: Problem):
        return [Individual(genome=genome, problem=problem) for genome in self.sample_pop(pop_size)]


class SeededPopInitializer(PopInitializer):
    @abstractmethod
    def get_seed(self, problem: Problem) -> Individual:
        pass


class UniformGlobalInitializer(PopInitializer):
    
    def prepare_sampler(self, _: dict):
        self.sampler = sample_uniform(self._bounds)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        return self.sampler(pop_size)


class LHSGlobalInitializer(PopInitializer):

    def prepare_sampler(self, _: dict):
        self.sampler = LatinHypercube(d=len(self._bounds))

    def sample_pop(self, pop_size: int) -> np.ndarray:
        sample = self.sampler.random(pop_size)
        return linear_scale(sample, self._bounds)


class SobolGlobalInitializer(PopInitializer):

    def prepare_sampler(self, _: dict):
        self.sampler = Sobol(d=len(self._bounds), scramble=True)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        sample = self.sampler.random(pop_size)
        return linear_scale(sample, self._bounds)


class GaussianInitializer(PopInitializer):

    def prepare_sampler(self, context: dict):
        try:
            seed_genome: np.ndarray = context["seed_genome"]
            std_dev: float = context["std_dev"]
        except KeyError:
            raise ValueError("GaussianInitializer requires a seed genome and a standard deviation")
        self.sampler = sample_normal(seed_genome, std_dev, self._bounds)

    def sample_pop(self, pop_size: int) -> np.ndarray:
        return self.sampler(pop_size)


class GaussianInitializerWithSeedInject(SeededPopInitializer):
    
    def prepare_sampler(self, context: dict):
        try:
            seed_ind: Individual = context["seed_ind"]
            std_dev: float = context["std_dev"]
            preserve_fitness: bool = context.get("preserve_fitness", True)
        except KeyError:
            raise ValueError("GaussianInitializer requires a seed individual and a standard deviation")
        self.sampler = sample_normal(seed_ind.genome, std_dev, self._bounds)
        self._preserve_fitness = preserve_fitness
        self._seed_ind = seed_ind

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
    
    def prepare_sampler(self, context: dict = None):
        self.injected_population: list[Individual] = context["injected_pop"]
        self._preserve_fitness: bool = context.get("preserve_fitness", True)

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
