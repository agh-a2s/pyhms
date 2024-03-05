from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyhms.demes.abstract_deme import AbstractDeme


class LocalStopCondition(ABC):
    @abstractmethod
    def __call__(self, deme: "AbstractDeme") -> bool:
        raise NotImplementedError()


class FitnessSteadiness(LocalStopCondition):
    """
    LSC is true if the average fitness of the last n_metaepochs is within max_deviation of the minimum fitness.
    """

    def __init__(self, max_deviation: float = 0.001, n_metaepochs: int = 5) -> None:
        self.max_deviation = max_deviation
        self.n_metaepochs = n_metaepochs

    def __call__(self, deme: "AbstractDeme") -> bool:
        if self.n_metaepochs > deme.metaepoch_count:
            return False

        avg_fits = [
            np.mean([ind.fitness for generation in deme._history[n] for ind in generation])
            for n in range(-self.n_metaepochs, 0)
        ]
        return np.mean(avg_fits) - np.min(avg_fits) <= self.max_deviation

    def __str__(self) -> str:
        return f"FitnessSteadiness(max_dev={self.max_deviation}, n={self.n_metaepochs})"


class AllChildrenStopped(LocalStopCondition):
    """
    LSC is true if all children of the deme are stopped.
    """

    def __call__(self, deme: "AbstractDeme") -> bool:
        if not deme.children:
            return False

        return all(not child.is_active for child in deme.children)
