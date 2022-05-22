"""
    Deme data.
"""
import numpy as np

from ..deme import AbstractDeme, EA_Deme
from .solution import Solution
from ..util import compute_centroid

class DemeData(AbstractDeme):
    def __init__(self, deme: EA_Deme) -> None:
        super().__init__(deme.id, deme.started_at)
        self._history = [Solution.simplify_population(pop) for pop in deme.history]

    @property
    def history(self) -> list:
        return self._history

    @property
    def centroid(self) -> np.array:
        return compute_centroid(self._history[-1])
