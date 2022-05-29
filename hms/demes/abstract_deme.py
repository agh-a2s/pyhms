from abc import ABC, abstractmethod
import numpy as np

from ..config import BaseLevelConfig
from ..util import compute_avg_fitness


class AbstractDeme(ABC):
    def __init__(self, id: str, started_at: int = 0, config: BaseLevelConfig = None) -> None:
        super().__init__()
        self._id = id
        self._started_at = started_at
        self._config = config
        self._lsc = config.lsc
        self._problem = config.problem
        self._bounds = config.bounds
        self._active = True

    @property
    def id(self) -> str:
        return self._id

    @property
    def started_at(self) -> int:
        return self._started_at

    @property
    @abstractmethod
    def history(self) -> list:
        raise NotImplementedError()

    @property
    @abstractmethod
    def centroid(self) -> np.array:
        raise NotImplementedError()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def all_individuals(self) -> list:
        inds = []
        for pop in self.history:
            inds += pop

        return inds

    @property
    def metaepoch_count(self) -> int:
        return len(self.history) - 1

    @property
    def best(self):
        return max(self.history[-1])

    @property
    def config(self):
        return self._config

    def avg_fitness(self, metaepoch=-1) -> float:
        return compute_avg_fitness(self.history[metaepoch])
