from abc import ABC, abstractmethod

from leap_ec.individual import Individual
from leap_ec.representation import Representation
import leap_ec.ops as lops
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from toolz import pipe

class AbstractEA(ABC):
    def __init__(self, problem, bounds, pop_size) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.pop_size = pop_size

    @abstractmethod
    def run(self, parents=None):
        raise NotImplementedError()

class SimpleEA(AbstractEA):
    """
    A simple single population EA (SEA skeleton).
    """
    def __init__(self, generations, problem, bounds, pop_size, pipeline, 
        k_elites=1, representation=None) -> None:
        super().__init__(problem, bounds, pop_size)
        self.generations = generations
        self.pipeline = pipeline
        self.k_elites = k_elites
        if representation is not None:
            self.representation = representation
        else:
            self.representation = Representation(initialize=create_real_vector(bounds=bounds))

    def run(self, parents=None):
        if parents is None:
            parents = self.representation.create_population(
                pop_size=self.pop_size, 
                problem=self.problem
                )
            parents = Individual.evaluate_population(parents)
        else:
            assert(self.pop_size == len(parents))

        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))

class SEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """
    def __init__(self, generations, problem, bounds, pop_size, mutation_std=1.0, 
        k_elites=1, representation=None) -> None:

        super().__init__(
            generations, 
            problem, 
            bounds, 
            pop_size, 
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                mutate_gaussian(std=mutation_std, hard_bounds=bounds, expected_num_mutations='isotropic'),
                lops.evaluate,
                lops.pool(size=pop_size)
            ], 
            k_elites=k_elites, 
            representation=representation
            )

    @classmethod
    def create(cls, generations, problem, bounds, pop_size, **kwargs):
        mutation_std = kwargs.get('mutation_std') or 1.0
        k_elites = kwargs.get('k_elites') or 1
        return cls(
            generations=generations,
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            mutation_std=mutation_std,
            k_elites=k_elites
            )
