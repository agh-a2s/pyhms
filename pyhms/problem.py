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

class EvalCutoffProblem(EvalCountingProblem):
    def __init__(self, decorated_problem: Problem, eval_cutoff: int):
        super().__init__(decorated_problem)
        self._eval_cutoff = eval_cutoff

    def evaluate(self, phenome, *args, **kwargs):
        if self._n_evals >= self._eval_cutoff:
            return np.inf
        return super().evaluate(phenome, *args, **kwargs)

class PrecisionCutoffProblem(EvalCountingProblem):
    def __init__(self, decorated_problem: Problem, global_optima: float, precision: float):
        super().__init__(decorated_problem)
        self._global_optima = global_optima
        self.precision = precision
        self.ETA = np.inf
        self.hit_precision = False

    def evaluate(self, phenome, *args, **kwargs):
        fitness = self._inner.evaluate(phenome, *args, **kwargs)
        if fitness -  - self._global_optima <= self.precision and not self.hit_precision:
            self.ETA = self._n_evals
            self.hit_precision = True
        return fitness

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