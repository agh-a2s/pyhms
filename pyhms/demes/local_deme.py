from leap_ec import Individual
from scipy import optimize as sopt
from structlog.typing import FilteringBoundLogger

from ..config import LocalOptimizationConfig
from .abstract_deme import AbstractDeme


class LocalDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: LocalOptimizationConfig,
        logger: FilteringBoundLogger,
        sprout_seed: Individual,
        started_at=0,
        parent_deme: AbstractDeme | None = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at)
        self._method = config.method
        self._sprout_seed = sprout_seed
        self._n_evals = 0
        starting_pop = [self._sprout_seed]
        self._history.append([starting_pop])
        self._run_history: list[Individual] = []

        self._options = {}
        if "maxiter" in config.__dict__:
            self._options["maxiter"] = config.__dict__["maxiter"]

    def run_metaepoch(self, _) -> None:
        x0 = self._sprout_seed.genome
        fun = self._problem.evaluate

        result = sopt.minimize(
            fun,
            x0,
            method=self._method,
            bounds=self._bounds,
            callback=self._history_callback,
            options=self._options,
        )

        # Accessing the result object gives the exact number of function evaluations.
        # Callback does not include jacobian approximation etc
        self._n_evals += result.nfev
        # Encapsulating all iterations in a list to match actual metaepoch count
        self._history.append([self._run_history])
        # By design local optimization is a one-metaepoch process
        self._active = False
        self.log("Local Deme run executed")

    @property
    def n_evaluations(self) -> int:
        return self._n_evals

    def _history_callback(self, intermediate_result) -> None:
        ind = Individual(intermediate_result.x, problem=self._problem)
        ind.fitness = intermediate_result.fun
        self._run_history.append(ind)

    def log(self, message: str) -> None:
        self._logger.info(
            message,
            id=self._id,
            best_fitness=self.best_current_individual.fitness,
            best_individual=self.best_current_individual.genome,
            n_evals=self.n_evaluations,
        )
