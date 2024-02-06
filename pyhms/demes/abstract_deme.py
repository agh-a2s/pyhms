from abc import ABC, abstractmethod

from leap_ec.individual import Individual
from pyhms.config import BaseLevelConfig
from pyhms.utils.misc_util import compute_centroid
from structlog.typing import FilteringBoundLogger


class AbstractDeme(ABC):
    def __init__(
        self,
        id: str,
        level: int,
        config: BaseLevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        seed: Individual = None,
    ) -> None:
        super().__init__()
        self._id = id
        self._started_at = started_at
        self._seed = seed
        self._level = level
        self._config = config
        self._lsc = config.lsc
        self._problem = config.problem
        self._bounds = config.bounds
        self._active = True
        self._centroid = None
        self._history = []
        self._children = []
        self._logger = logger

        # Additional low-level options
        self._hibernating = False

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
    def centroid(self) -> Individual:
        if self._centroid is None:
            self._centroid = compute_centroid(self.current_population)
        return self._centroid

    @property
    def history(self) -> list:
        return self._history

    @property
    def all_individuals(self) -> list:
        return [ind for pop in self.history for ind in pop]

    @property
    def n_evaluations(self) -> int:
        return len(self.all_individuals)

    @property
    def current_population(self) -> list:
        return self._history[-1]

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
    def config(self):
        return self._config

    @property
    def children(self):
        return self._children

    def add_child(self, deme):
        self._children.append(deme)

    @abstractmethod
    def run_metaepoch(self, tree):
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
        if self._seed is None:
            return f"Root deme {self.id} with best achieved fitness {best_fitness}"
        else:
            return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._seed.genome} with best {best_fitness}"
