import time

import numpy as np
from leap_ec.problem import FunctionProblem, Problem


class EvalCountingProblem(Problem):
    """
    A decorator for a leap_ec.Problem instance that counts the number of evaluations performed.

    This class wraps around any instance of `Problem` and counts how many times the
    `evaluate` method is called. This is useful for monitoring and limiting the computational
    cost of optimization processes.

    Example:
        >>> from leap_ec.problem import FunctionProblem
        >>> from pyhms.problem import EvalCountingProblem
        >>> problem = FunctionProblem(lambda x: -x**2, maximize=True)
        >>> counting_problem = EvalCountingProblem(problem)
        >>> counting_problem.evaluate(np.array([2]))
        >>> print(counting_problem.n_evaluations)
    """

    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals = 0

    def evaluate(self, phenome, *args, **kwargs):
        ret_val = self._inner.evaluate(phenome, *args, **kwargs)
        self._n_evals += 1
        return ret_val

    def worse_than(self, first_fitness, second_fitness):
        return self._inner.worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._inner.equivalent(first_fitness, second_fitness)

    @property
    def n_evaluations(self):
        return self._n_evals

    def __str__(self) -> str:
        if isinstance(self._inner, FunctionProblem):
            inner_str = f"FunctionProblem({self._inner.__dict__})"
        else:
            inner_str = str(self._inner)
        return f"EvalCountingProblem({inner_str})"


class EvalCutoffProblem(EvalCountingProblem):
    def __init__(self, decorated_problem: Problem, eval_cutoff: int):
        super().__init__(decorated_problem)
        self._eval_cutoff = eval_cutoff

    def evaluate(self, phenome, *args, **kwargs):
        if self._n_evals >= self._eval_cutoff:
            return np.inf
        return super().evaluate(phenome, *args, **kwargs)


class PrecisionCutoffProblem(EvalCountingProblem):
    def __init__(
        self, decorated_problem: Problem, global_optima: float, precision: float
    ):
        super().__init__(decorated_problem)
        self._global_optima = global_optima
        self.precision = precision
        self.ETA = np.inf
        self.hit_precision = False

    def evaluate(self, phenome, *args, **kwargs):
        fitness = self._inner.evaluate(phenome, *args, **kwargs)
        if fitness - -self._global_optima <= self.precision and not self.hit_precision:
            self.ETA = self._n_evals
            self.hit_precision = True
        return fitness


class StatsGatheringProblem(Problem):
    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals = 0
        self._durations = []

    def evaluate(self, phenome, *args, **kwargs) -> float:
        start_time = time.perf_counter()
        ret_val = self._inner.evaluate(phenome, *args, **kwargs)
        end_time = time.perf_counter()
        self._n_evals += 1
        self._durations.append(end_time - start_time)
        return ret_val

    def worse_than(self, first_fitness, second_fitness):
        return self._inner.worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._inner.equivalent(first_fitness, second_fitness)

    @property
    def n_evaluations(self) -> int:
        return self._n_evals

    @property
    def durations(self) -> list[float]:
        return self._durations

    @property
    def duration_stats(self) -> tuple[float, float]:
        return np.mean(self._durations), np.std(self._durations)

    def __str__(self) -> str:
        if isinstance(self._inner, FunctionProblem):
            inner_str = f"FunctionProblem({self._inner.__dict__})"
        else:
            inner_str = str(self._inner)
        return f"StatsGatheringProblem({inner_str})"
