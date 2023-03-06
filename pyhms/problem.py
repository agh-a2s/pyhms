import time
import numpy as np
from leap_ec.problem import FunctionProblem, Problem

class EvalCountingProblem(Problem):
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

class StatsGatheringProblem(Problem):
    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals = 0
        self._durations = []

    def evaluate(self, phenome, *args, **kwargs):
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
        if isinstance(self._inner, FunctionProblem):
            inner_str = f"FunctionProblem({self._inner.__dict__})"
        else:
            inner_str = str(self._inner)
        return f"StatsGatheringProblem({inner_str})"