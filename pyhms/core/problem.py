import time
import numpy as np
from pymoo.core.problem import Problem

class EvalCountingProblem(Problem):
    def __init__(self, fit_fun, n_var, xl, xu):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.fit_fun = fit_fun
        self._n_evals = 0

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([self.fit_fun(ind) for ind in x])
        self._n_evals += 1

    @property
    def n_evaluations(self):
        return self._n_evals

    def __str__(self) -> str:
        return f"EvalCountingProblem({super().__str__()})"


class StatsGatheringProblem(Problem):
    def __init__(self, fit_fun, n_var, xl, xu):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.fit_fun = fit_fun
        self._n_evals = 0
        self._durations = []

    def _evaluate(self, x, out, *args, **kwargs):
        start_time = time.perf_counter()
        out["F"] = np.array([self.fit_fun(ind) for ind in x])
        end_time = time.perf_counter()
        self._durations.append(end_time - start_time)
        self._n_evals += 1

    @property
    def n_evaluations(self):
        return self._n_evals

    @property
    def durations(self):
        return self._durations

    @property
    def duration_stats(self):
        m = np.mean(self._durations)
        s = np.sqrt(np.mean((self._durations - m) ** 2))
        return m, s

    def __str__(self) -> str:
        return f"StatsGatheringProblem({super().__str__()})"