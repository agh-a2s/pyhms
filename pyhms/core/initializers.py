from abc import ABC, abstractmethod
from pyhms.core.individual import Individual
from pyhms.core.problem import Problem
from pyhms.utils.samplers import sample_normal, sample_uniform
import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol

class PopInitializer(ABC):

    def __init__(self, problem: Problem, bounds: np.ndarray | None = None):
        self._problem = problem
        self._bounds = bounds

    @abstractmethod
    def sample_pop(self, pop_size: int | None) -> list[Individual]:
        pass

    def __call__(self, pop_size):
        return self.sample_pop(pop_size)

class SeededPopInitializer(PopInitializer):

    @abstractmethod
    def get_seed(self) -> Individual:
        pass

class UniformGlobalInitializer(PopInitializer):

    def __init__(self, problem: Problem, bounds: np.ndarray):
        super().__init__(problem, bounds)
        self.sampler = sample_uniform(bounds)

    def sample_pop(self, pop_size: int) -> list[Individual]:
        return [Individual(genome=genome, problem=self._problem) for genome in self.sampler(pop_size)]

class LHSGlobalInitializer(PopInitializer):

    def __init__(self, problem: Problem, bounds: np.ndarray, random_seed: int = None):
        super().__init__(problem, bounds)
        self.sampler = LatinHypercube(d=len(bounds), seed=random_seed)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]

    def sample_pop(self, pop_size: int) -> list[Individual]:
        sample = self.sampler.random(pop_size)
        genomes = self.lower_bounds + sample * (self.upper_bounds - self.lower_bounds)
        return [Individual(genome, problem=self._problem) for genome in genomes]

class SobolGlobalInitializer(PopInitializer):

    def __init__(self, problem: Problem, bounds: np.ndarray, random_seed: int = None):
        super().__init__(problem, bounds)
        self.sampler = Sobol(d=len(bounds), scramble=True, seed=random_seed)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]

    def sample_pop(self, pop_size: int) -> list[Individual]:
        sample = self.sampler.random(pop_size)
        genomes = self.lower_bounds + sample * (self.upper_bounds - self.lower_bounds)
        return [Individual(genome, problem=self._problem) for genome in genomes]

class GaussianInitializer(PopInitializer):

    def __init__(self, seed: np.ndarray, std_dev: float, problem: Problem, bounds: np.ndarray | None = None):
        super().__init__(problem, bounds)
        self.sampler = sample_normal(seed, std_dev, bounds)
        self._seed = seed

    def sample_pop(self, pop_size: int) -> list[Individual]:
        return [Individual(genome=genome, problem=self._problem) for genome in self.sampler(pop_size)]

class GaussianInitializerWithSeedInject(SeededPopInitializer):

    def __init__(self, seed: Individual, std_dev: float, problem: Problem, bounds: np.ndarray | None = None):
        super().__init__(problem, bounds)
        self.sampler = sample_normal(seed.genome, std_dev, bounds)
        if seed.problem == problem:
            self._seed_ind = seed
        else:
            self._seed_ind = Individual(genome=seed.genome, problem=problem)

    def sample_pop(self, pop_size: int) -> list[Individual]:
        return [Individual(genome=genome, problem=self._problem) for genome in self.sampler(pop_size-1)] + [self._seed_ind]

    def get_seed(self) -> Individual:
        return self._seed_ind

class InjectionInitializer(SeededPopInitializer):

    def __init__(self, injected_population: list[Individual], problem: Problem, bounds: np.ndarray | None = None):
        super().__init__(problem, bounds)
        if injected_population[0].problem == problem:
            self.injected_population = injected_population
        else:
            self.injected_population = [Individual(genome=ind.genome, problem=problem) for ind in injected_population]

    def sample_pop(self, pop_size: int | None) -> list[Individual]:
        if pop_size is None:
            return self.injected_population
        else:
            if any(ind.fitness is None for ind in self.injected_population):
                return np.random.choice(self.injected_population, pop_size, replace=False).tolist()
            else:
                return sorted(self.injected_population, reverse=True)[:pop_size]
    
    def get_seed(self) -> Individual:
        return self.sample_pop(1)[0]