"""
    Deme data.
"""

import numpy as np

from ...demes.abstract_deme import AbstractDeme
from ..misc_util import compute_centroid
from .pers_solution import Solution


class DemeData(AbstractDeme):
    def __init__(self, deme: AbstractDeme) -> None:
        super().__init__(deme.id, deme.started_at, deme.config, logger=deme._logger)
        self._history = [Solution.simplify_population(pop) for pop in deme.history]

    @property
    def history(self) -> list:
        return self._history

    @property
    def centroid(self) -> np.ndarray:
        return compute_centroid(self._history[-1])
