from cma import CMAEvolutionStrategy
import numpy as np
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder

from .abstract_deme import AbstractDeme
from ..config import CMALevelConfig
from ..loggers import deme_logger
from ..util import compute_centroid


class CMADeme(AbstractDeme):

    def __init__(self, id: str, config: CMALevelConfig, x0, started_at=0, leaf=False) -> None:
        super().__init__(id, started_at, config)
        self._x0 = x0
        self._sigma0 = config.sigma0
        self._cma_es = CMAEvolutionStrategy(x0.genome, config.sigma0)

        self._centroid = None
        self._history = [[self._x0]]
        self._active = True
        self._children = []
        self._leaf = leaf

    @property
    def history(self) -> list:
        return self._history

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self._history[-1])
        return self._centroid

    def run_metaepoch(self) -> None:
        solutions = self._cma_es.ask()
        individuals = [Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in solutions]
        self._cma_es.tell(solutions, [ind.evaluate() for ind in individuals])
        self._cma_es.disp()
        self._centroid = None

        self._history.append(individuals)

        if self._lsc(self):
            self._active = False
            deme_logger.debug(f"{self} stopped after {self.metaepoch_count} metaepochs")

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = max(self._history[-1])
        return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._x0.genome} with best {bsf}"

