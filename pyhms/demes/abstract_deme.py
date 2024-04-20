from abc import ABC, abstractmethod

import numpy as np
from pyhms.config import BaseLevelConfig
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual
from ..core.problem import Problem
from ..problem import EvalCountingProblem
from ..stop_conditions import LocalStopCondition, UniversalStopCondition


def compute_centroid(population: list[Individual]) -> np.ndarray:
    return np.mean([ind.genome for ind in population], axis=0)


class AbstractDeme(ABC):
    def __init__(
        self,
        id: str,
        level: int,
        config: BaseLevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual = None,
    ) -> None:
        super().__init__()
        self._id = id
        self._started_at = started_at
        self._sprout_seed = sprout_seed
        self._level = level
        self._config: BaseLevelConfig = config
        self._lsc: LocalStopCondition | UniversalStopCondition = config.lsc
        self._problem: EvalCountingProblem = EvalCountingProblem(config.problem)
        self._bounds: np.ndarray = config.bounds
        self._active: bool = True
        self._centroid: np.ndarray | None = None
        # History of populations is a nested list, where each element is a list of individuals.
        # The reason for this is that we want to keep track of the entire history of the deme,
        # and for some algorithms (e.g. CMA-ES) HMS can run multiple generations during one metaepoch.
        self._history: list[list[list[Individual]]] = []
        self._children: list[AbstractDeme] = []
        self._logger: FilteringBoundLogger = logger

        # Additional low-level options
        self._hibernating: bool = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def started_at(self) -> int:
        return self._started_at

    @property
    def level(self) -> int:
        return self._level

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            self._centroid = compute_centroid(self.current_population)
        return self._centroid

    @property
    def history(self) -> list[list[Individual]]:
        # Returns flattened self._history = list of all generations.
        return [generation for metaepoch_generations in self._history for generation in metaepoch_generations]

    @property
    def all_individuals(self) -> list:
        return [ind for pop in self.history for ind in pop]

    @property
    def n_evaluations(self) -> int:
        return self._problem.n_evaluations

    @property
    def current_population(self) -> list[Individual]:
        return self.history[-1]

    @property
    def best_current_individual(self) -> Individual:
        return max(self.current_population)

    @property
    def best_individual(self) -> Individual:
        return max(self.all_individuals)

    @property
    def metaepoch_count(self) -> int:
        return len(self._history) - 1

    @property
    def config(self) -> BaseLevelConfig:
        return self._config

    @property
    def children(self) -> list["AbstractDeme"]:
        return self._children

    def add_child(self, deme: "AbstractDeme") -> None:
        self._children.append(deme)

    @abstractmethod
    def run_metaepoch(self, tree) -> None:
        raise NotImplementedError()

    def log(self, message: str) -> None:
        self._logger.info(
            message,
            id=self._id,
            best_fitness=self.best_current_individual.fitness,
            best_individual=self.best_current_individual.genome,
            n_evals=self.n_evaluations,
            centroid=self.centroid,
        )

    def __str__(self) -> str:
        best_fitness = self.best_current_individual.fitness
        if self._sprout_seed is None:
            return f"Root deme {self.id} with best achieved fitness {best_fitness}"
        else:
            return f"""Deme {self.id}, metaepoch {self.started_at} and
            seed {self._sprout_seed.genome} with best {best_fitness}"""

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} {self.id}"

    @property
    def best_fitness_by_metaepoch(self) -> dict[int, float]:
        metaepoch_to_best_fitness = {}
        for metaepoch_idx, metaepoch_history in enumerate(self._history):
            if not metaepoch_history:
                continue
            best_fitness = max(pop for generation in metaepoch_history for pop in generation).fitness
            metaepoch_to_best_fitness[metaepoch_idx + self._started_at] = best_fitness
        return metaepoch_to_best_fitness
