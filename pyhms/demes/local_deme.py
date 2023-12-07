from leap_ec import Individual
from scipy import optimize as sopt

from .abstract_deme import AbstractDeme
from ..config import LocalOptimizationConfig

class LocalDeme(AbstractDeme):

    def __init__(self, id: str, level: int, config: LocalOptimizationConfig, x0: Individual, started_at=0) -> None:
        super().__init__(id, level, config, started_at)
        self._method = config.method
        self._maxfun = config.maxfun
        self._x0 = x0

    def run_metaepoch(self, tree) -> None:
        x0 = self._x0.genome
        fun = self._problem.evaluate

        sopt.minimize(fun, x0, method=self._method, bounds=self._bounds, options={'maxfun': self._maxfun}, callback=self._history_callback)

    def _history_callback(self, intermediate_result) -> None:
        self.history.append(Individual(intermediate_result.x, problem=self._problem))