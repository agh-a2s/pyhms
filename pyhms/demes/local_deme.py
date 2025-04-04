from scipy import optimize as sopt

from ..config import LocalOptimizationConfig
from ..core.individual import Individual
from .abstract_deme import AbstractDeme, DemeInitArgs


class LocalDeme(AbstractDeme):
    def __init__(
        self,
        deme_init_args: DemeInitArgs,
    ) -> None:
        super().__init__(deme_init_args)
        config: LocalOptimizationConfig = deme_init_args.config  # type: ignore[assignment]
        self._method = config.method
        self._sprout_seed = deme_init_args.sprout_seed
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
