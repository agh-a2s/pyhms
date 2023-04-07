"""
    Global stopping conditions.
"""
import logging

from abc import ABC, abstractmethod
from typing import Any, Union, List

from ..problem import StatsGatheringProblem, EvalCountingProblem, PrecisionCutoffProblem
from ..tree import DemeTree

logger = logging.getLogger(__name__)


class gsc(ABC):
    @abstractmethod
    def satisfied(self, tree: DemeTree) -> bool:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.satisfied(*args, **kwds)


class root_stopped(gsc):
    def satisfied(self, tree: DemeTree) -> bool:
        return not tree.root.active

    def __str__(self) -> str:
        return "root_stopped"


class all_stopped(gsc):
    def satisfied(self, tree: DemeTree) -> bool:
        return len(list(tree.active_demes)) == 0

    def __str__(self) -> str:
        return "all_stopped"


class fitness_eval_limit_reached(gsc):
    def __init__(self, limit: int, weights: Union[List[float], str] = 'equal') -> None:
        super().__init__()
        self.limit = limit
        self.weights = weights

    def satisfied(self, tree: DemeTree) -> bool:
        levels = tree.config.levels
        n_levels = len(levels)
        if self.weights is None or isinstance(self.weights, str):
            self._transform_weights(n_levels)

        n_evals = 0
        for i in range(n_levels):
            if not isinstance(levels[i].problem, StatsGatheringProblem) and not isinstance(levels[i].problem, EvalCountingProblem):
                raise ValueError("Problem has to be an instance of EvalCountingProblem")

            n_evals += self.weights[i] * levels[i].problem.n_evaluations

        return n_evals >= self.limit

    def _transform_weights(self, n_levels: int):
        if self.weights == 'root':
            self.weights = [0 for _ in range(n_levels)]
            self.weights[0] = 1
        elif self.weights == 'equal' or self.weights is None:
            self.weights = [1 for _ in range(n_levels)]

    def __str__(self) -> str:
        return f"fitness_eval_limit_reached(limit={self.limit}, weights={self.weights})"

class singular_problem_eval_limit_reached(gsc):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
    
    def satisfied(self, tree: DemeTree) -> bool:
        problem = tree.root._problem
        if not isinstance(problem, StatsGatheringProblem) and not isinstance(problem, EvalCountingProblem):
            raise ValueError("Problem has to be an instance of EvalCountingProblem")
        return problem.n_evaluations >= self.limit
    
    def __str__(self) -> str:
        return f"singular_problem_eval_limit_reached(limit={self.limit})"
    
class singular_problem_precision_reached(gsc):
    def __init__(self, problem: PrecisionCutoffProblem):
        super().__init__()
        self.problem = problem
    
    def satisfied(self, _: DemeTree) -> bool:
        return self.problem.hit_precision
    
    def __str__(self) -> str:
        return f"singular_problem_precision_reached(precision={self.problem.precision})"


class no_active_nonroot_demes(gsc):
    def __init__(self, n_metaepochs: int = 5) -> None:
        super().__init__()
        self.n_metaepochs = n_metaepochs

    def satisfied(self, tree: DemeTree) -> bool:
        step = tree.metaepoch_count
        logger.debug(f"Step {step}")
        for level_no in range(1, tree.height):
            if len(tree.levels[level_no]) == 0:
                return False

            for deme in tree.levels[level_no]:
                logger.debug(f"Deme {deme.id} st {deme.started_at} mc {deme.metaepoch_count}")
                if deme.active or \
                        step <= deme.started_at + deme.metaepoch_count + self.n_metaepochs:
                    return False

        return True

    def __str__(self) -> str:
        return f"no_active_nonroot_demes({self.n_metaepochs})"
