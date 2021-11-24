import statistics as stats
import time
import numpy as np
from leap_ec.problem import Problem

def square(x) -> float:
    if not isinstance(x, np.ndarray):
        xa = np.array(x)
    if xa.ndim == 0:
        xa = np.array([x])
    return sum(xa**2)

class StatsGatheringProblem(Problem):
    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals = 0
        self._durations = []

    def evaluate(self, phenome, *args, **kwargs):
        start_time = time.time()
        ret_val = self._inner.evaluate(phenome, *args, **kwargs)
        end_time = time.time()
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
        m = stats.mean(self._durations)
        s = stats.stdev(self._durations, xbar=m)
        return m, s
