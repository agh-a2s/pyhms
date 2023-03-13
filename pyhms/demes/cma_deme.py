import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from scipy import optimize as sopt

from .abstract_deme import AbstractDeme
from .deme_config import CMALevelConfig
from ..utils.misc_util import compute_centroid
from ..operators.callbacks import HistoryCallback


class CMADeme(AbstractDeme):

    def __init__(self, id: str, config: CMALevelConfig, x0, started_at=0) -> None:
        super().__init__(id, config, started_at, False)
        self._x0 = x0

        self._cma_es = CMAES(x0.get("X"), config.sigma0)
        self._cma_es.setup(self._problem, callback=HistoryCallback())

        self._centroid = None
        self._active = True
        self._children = []

    @property
    def algorithm(self):
        return self._cma_es

    @property
    def history(self) -> list:
        return self._cma_es.callback.data["history"]

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self.history()[-1])
        return self._centroid

    def run_metaepoch(self) -> None:
        epoch_counter = 0
        while epoch_counter < self._config.generations:
            self._cma_es.next()
            epoch_counter += 1

        self._centroid = None

        if self._lsc(self) or not self._cma_es.has_next():
            self._active = False

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = self.best
        return f'Deme {self.id}, metaepoch {self.started_at} and seed {self._x0.get("X")} with best {bsf.get("F")}'

