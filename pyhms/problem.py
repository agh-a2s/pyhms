import time

import numpy as np
from leap_ec.problem import FunctionProblem, Problem


class EvalCountingProblem(Problem):
    """
    A decorator for a leap_ec.Problem instance that counts the number of evaluations performed.

    This class wraps around any instance of `Problem` and counts how many times the
    `evaluate` method is called. This is useful for monitoring and limiting the computational
    cost of optimization processes.

    :param leap_ec.Problem decorated_problem: The problem to be decorated.

    .. code-block:: python

        >>> from leap_ec.problem import FunctionProblem
        >>> from pyhms.problem import EvalCountingProblem
        >>> import numpy as np
        >>> problem = FunctionProblem(lambda x: -x**2, maximize=True)
        >>> counting_problem = EvalCountingProblem(problem)
        >>> counting_problem.evaluate(2.0)
        >>> print(counting_problem.n_evaluations)
        1
    """

    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals: int = 0

    def evaluate(self, phenome, *args, **kwargs):
        ret_val = self._inner.evaluate(phenome, *args, **kwargs)
        self._n_evals += 1
        return ret_val

    def worse_than(self, first_fitness, second_fitness):
        return self._inner.worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._inner.equivalent(first_fitness, second_fitness)

    @property
    def n_evaluations(self) -> int:
        return self._n_evals

    def __str__(self) -> str:
        if isinstance(self._inner, FunctionProblem):
            inner_str = f"FunctionProblem({self._inner.__dict__})"
        else:
            inner_str = str(self._inner)
        return f"EvalCountingProblem({inner_str})"


class EvalCutoffProblem(EvalCountingProblem):
    """
    A decorator for a leap_ec.Problem instance that imposes a cutoff on the number of evaluations.

    This class extends `EvalCountingProblem` by adding a functionality to stop evaluations
    once a specified cutoff limit is reached. Evaluations beyond this limit will return a
    predefined value, effectively simulating an infinite cost or penalty. This can be useful
    in scenarios where computational resources are limited or when implementing certain
    types of optimization algorithms that require evaluation limits.

    Note:
        It returns `np.inf` for evaluations beyond the cutoff if the problem is a minimization one,
        and `-np.inf` for maximization.

    .. code-block:: python

        >>> from leap_ec.problem import FunctionProblem
        >>> from pyhms.problem import EvalCutoffProblem
        >>> import numpy as np
        >>> problem = FunctionProblem(lambda x: -x**2, maximize=True)
        >>> cutoff_problem = EvalCutoffProblem(problem, eval_cutoff=1)
        >>> cutoff_problem.evaluate(2.0)
        >>> cutoff_problem.evaluate(1.0)
        inf

    """

    def __init__(self, decorated_problem: Problem, eval_cutoff: int):
        super().__init__(decorated_problem)
        self._eval_cutoff = eval_cutoff

    def evaluate(self, phenome, *args, **kwargs):
        if self._n_evals >= self._eval_cutoff:
            return -np.inf if self._inner.maximize else np.inf
        return super().evaluate(phenome, *args, **kwargs)


class PrecisionCutoffProblem(EvalCountingProblem):
    """
    A decorator for a Problem instance that introduces a precision-based cutoff criterion.

    The class extends `EvalCountingProblem` and tracks whether the specified precision
    relative to the global optimum has been achieved and records the evaluation count (ETA)
    when this occurs for the first time.

    This class is useful for optimization problems where the goal is to find a solution that
    is close enough to the known global optimum within a certain precision threshold.

    Example:
        >>> from leap_ec.problem import FunctionProblem
        >>> from pyhms.problem import PrecisionCutoffProblem
        >>> import numpy as np
        >>> problem = FunctionProblem(lambda x: -x**2, maximize=True)
        >>> precision_cutoff_problem = PrecisionCutoffProblem(problem, 0, 1e-4)
        >>> precision_cutoff_problem.evaluate(0)
        >>> print(precision_cutoff_problem.ETA)
        0
    """

    def __init__(self, decorated_problem: Problem, global_optima: float, precision: float):
        super().__init__(decorated_problem)
        self._global_optima = global_optima
        self.precision = precision
        self.ETA = np.inf
        self.hit_precision = False

    def evaluate(self, phenome, *args, **kwargs):
        fitness = super().evaluate(phenome, *args, **kwargs)
        if abs(fitness - self._global_optima) <= self.precision and not self.hit_precision:
            self.ETA = self._n_evals
            self.hit_precision = True
        return fitness


class StatsGatheringProblem(Problem):
    """
    A decorator for a leap_ec.Problem instance that gathers statistics about evaluation times.

    This class wraps around any instance of `Problem` to record the duration of each
    evaluation performed on the decorated problem. It's particularly useful for performance
    analysis, enabling the monitoring of how long each call to `evaluate` takes.

    Example:
        >>> from leap_ec.problem import FunctionProblem
        >>> from pyhms.problem import StatsGatheringProblem
        >>> import numpy as np
        >>> problem = FunctionProblem(lambda x: -x**2, maximize=True)
        >>> stats_gathering_problem = StatsGatheringProblem(problem)
        >>> stats_gathering_problem.evaluate(2.0)
        >>> print(stats_gathering_problem.durations)
        [1.2750009773299098e-05]

    """

    def __init__(self, decorated_problem: Problem):
        super().__init__()
        self._inner: Problem = decorated_problem
        self._n_evals = 0
        self._durations: list[float] = []

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
    def n_evaluations(self) -> int:
        return self._n_evals

    @property
    def durations(self) -> list[float]:
        return self._durations

    @property
    def duration_stats(self) -> tuple[np.float_, np.float_]:
        return np.mean(self._durations), np.std(self._durations)

    def __str__(self) -> str:
        if isinstance(self._inner, FunctionProblem):
            inner_str = f"FunctionProblem({self._inner.__dict__})"
        else:
            inner_str = str(self._inner)
        return f"StatsGatheringProblem({inner_str})"
