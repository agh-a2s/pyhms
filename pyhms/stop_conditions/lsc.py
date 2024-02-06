"""
    Local stopping conditions.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..demes.abstract_deme import AbstractDeme


class lsc(ABC):
    @abstractmethod
    def satisfied(self, deme: AbstractDeme) -> bool:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.satisfied(*args, **kwds)


class fitness_steadiness(lsc):
    def __init__(self, max_deviation: float = 0.001, n_metaepochs: int = 5) -> None:
        super().__init__()
        self.max_deviation = max_deviation
        self.n_metaepochs = n_metaepochs

    def satisfied(self, deme: AbstractDeme) -> bool:
        if self.n_metaepochs > deme.metaepoch_count:
            return False

        avg_fits = [np.mean([ind.fitness for ind in deme.history[n]]) for n in range(-self.n_metaepochs, 0)]
        return np.mean(avg_fits) - np.min(avg_fits) <= self.max_deviation

    def __str__(self) -> str:
        return f"fitness_steadiness(max_dev={self.max_deviation}, n={self.n_metaepochs})"


class all_children_stopped(lsc):
    def satisfied(self, deme: AbstractDeme) -> bool:
        ch = deme.children
        return not (ch == []) and np.all([not c.active for c in ch])  # type: ignore[return-value]

    def __str__(self) -> str:
        return "all_children_stopped"
