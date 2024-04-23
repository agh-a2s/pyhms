from abc import ABC, abstractmethod
from typing import Any

import leap_ec.ops as lops
from leap_ec.real_rep.ops import mutate_gaussian

from ...core.individual import Individual
from ...core.problem import Problem

DEFAULT_K_ELITES = 1
DEFAULT_GENERATIONS = 1
DEFAULT_MUTATION_STD = 1.0


def pipe(data, *funcs):
    for func in funcs:
        data = func(data)
    return data


class AbstractEA(ABC):
    def __init__(
        self,
        problem: Problem,
        pop_size: int,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = problem.bounds
        self.pop_size = pop_size

    @abstractmethod
    def run(self, parents: list[Individual] | None = None):
        raise NotImplementedError()

    @classmethod
    def create(cls, problem: Problem, pop_size: int, **kwargs):
        return cls(problem=problem, pop_size=pop_size, **kwargs)


class SimpleEA(AbstractEA):
    """
    A simple single population EA (SEA skeleton).
    """

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        pipeline: list[Any],
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
    ) -> None:
        super().__init__(problem, pop_size)
        self.generations = generations
        self.pipeline = pipeline
        self.k_elites = k_elites

    def run(self, parents: list[Individual] | None = None) -> list[Individual]:
        assert self.pop_size == len(parents)
        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))


class SEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        mutation_std: float | None = DEFAULT_MUTATION_STD,
    ) -> None:
        super().__init__(
            problem,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                mutate_gaussian(
                    std=mutation_std,
                    bounds=problem.bounds,
                    expected_num_mutations="isotropic",
                ),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
        )

    @classmethod
    def create(cls, problem: Problem, pop_size: int, **kwargs):
        return cls(
            problem=problem,
            pop_size=pop_size,
            generations=kwargs.get("generations", DEFAULT_GENERATIONS),
            k_elites=kwargs.get("k_elites", DEFAULT_K_ELITES),
            mutation_std=kwargs.get("mutation_std", DEFAULT_MUTATION_STD),
        )
