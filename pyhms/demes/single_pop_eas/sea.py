from abc import ABC, abstractmethod
from typing import Any

import leap_ec.ops as lops
import numpy as np
from leap_ec.individual import Individual
from leap_ec.problem import Problem
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from toolz import pipe

DEFAULT_K_ELITES = 1
DEFAULT_GENERATIONS = 1
DEFAULT_MUTATION_STD = 1.0


class AbstractEA(ABC):
    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.pop_size = pop_size

    @abstractmethod
    def run(self, parents: list[Individual] | None = None):
        raise NotImplementedError()

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(problem=problem, bounds=bounds, pop_size=pop_size, **kwargs)


class SimpleEA(AbstractEA):
    """
    A simple single population EA (SEA skeleton).
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        pipeline: list[Any],
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
    ) -> None:
        super().__init__(problem, bounds, pop_size)
        self.generations = generations
        self.pipeline = pipeline
        self.k_elites = k_elites
        if representation is not None:
            self.representation = representation
        else:
            self.representation = Representation(initialize=create_real_vector(bounds=bounds))

    def run(self, parents: list[Individual] | None = None) -> list[Individual]:
        if parents is None:
            parents = self.representation.create_population(pop_size=self.pop_size, problem=self.problem)
            parents = Individual.evaluate_population(parents)
        else:
            assert self.pop_size == len(parents)

        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))


class SEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
        mutation_std: float | None = DEFAULT_MUTATION_STD,
    ) -> None:
        super().__init__(
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations="isotropic"),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, problem: Problem, bounds: np.ndarray, pop_size: int, **kwargs):
        return cls(
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            generations=kwargs.get("generations", DEFAULT_GENERATIONS),
            k_elites=kwargs.get("k_elites", DEFAULT_K_ELITES),
            mutation_std=kwargs.get("mutation_std", DEFAULT_MUTATION_STD),
        )
