import numpy as np
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from structlog.typing import FilteringBoundLogger
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy import optimize as sopt

from ..config import CMALevelConfig
from .abstract_deme import AbstractDeme


class QuadraticSurrogateDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: CMALevelConfig,
        logger: FilteringBoundLogger,
        x0: Individual,
        started_at: int = 0,
        parent_deme: AbstractDeme | None = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, x0)
        X = np.array([ind.genome for pop in parent_deme.history for ind in pop])
        y = np.array([ind.fitness for pop in parent_deme.history for ind in pop])
        quadratic_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
        )
        quadratic_model.fit(X, y)
        self.surrogate_model = quadratic_model
        self._history.append([[self._sprout_seed]])

    def run_metaepoch(self, _) -> None:
        fun = lambda x: self.surrogate_model.predict(x.reshape(1, -1))[0]
        result = sopt.minimize(
            fun,
            self._sprout_seed.genome,
            method="L-BFGS-B",
            bounds=self._bounds,
        )
        individual = Individual(result.x, problem=self._problem)
        individual.evaluate()
        self._history.append([[individual]])
        self._active = False
