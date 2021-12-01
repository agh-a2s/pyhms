from leap_ec.individual import Individual
import numpy as np

from .sea import AbstractEA

class NullEA(AbstractEA):
    def __init__(self, problem, bounds, pop_size=1, seed=None) -> None:
        super().__init__(problem, bounds, pop_size)
        if seed is not None:
            x = np.asarray(seed)
        else:
            x = np.zeros(len(bounds))        
        self.population = [Individual(x, problem=problem)]
        Individual.evaluate_population(self.population)


    def run(self, parents=None):
        if parents is not None:
            self.population = parents
        return self.population

    @classmethod
    def create(cls, problem, bounds, pop_size, **kwargs):
        seed = kwargs.get('seed')
        return cls(problem, bounds, pop_size, seed)
