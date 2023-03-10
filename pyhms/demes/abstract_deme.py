from abc import ABC, abstractmethod
import numpy as np

from .deme_config import BaseLevelConfig
from ..utils.misc_util import compute_avg_fitness


class AbstractDeme(ABC):
    def __init__(self, id: str, config: BaseLevelConfig = None, started_at: int = 0, leaf: bool = False) -> None:
        super().__init__()
        self._id = id
        self._started_at = started_at
        self._config = config
        self._lsc = config.lsc
        self._problem = config.problem
        self._bounds = config.bounds
        self._active = True
        self._leaf = leaf

    @property
    def id(self) -> str:
        return self._id

    @property
    def started_at(self) -> int:
        return self._started_at

    @property
    @abstractmethod
    def algorithm(self):
        raise NotImplementedError()

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
    def is_leaf(self) -> bool:
        return self._leaf

    @property
    def best(self):
        return max(self.history[-1])

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
    def config(self):
        return self._config

    def avg_fitness(self, metaepoch=-1) -> float:
        return compute_avg_fitness(self.history[metaepoch])
