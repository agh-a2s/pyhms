"""
    Global stopping conditions.
"""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from ..problem import EvalCountingProblem, PrecisionCutoffProblem, StatsGatheringProblem
from ..tree import DemeTree

logger = logging.getLogger(__name__)


class GSC(ABC):
    @abstractmethod
    def satisfied(self, tree: DemeTree) -> bool:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> bool:
        return self.satisfied(*args, **kwds)


class RootStopped(GSC):
    """
    GSC is true if the root is not active.
    """

    def satisfied(self, tree: DemeTree) -> bool:
        return not tree.root.is_active

    def __str__(self) -> str:
        return "RootStopped"


class AllStopped(GSC):
    """
    GSC is true if there are no active demes in the tree.
    """

    def satisfied(self, tree: DemeTree) -> bool:
        return len(list(tree.active_demes)) == 0

    def __str__(self) -> str:
        return "AllStopped"


class WeightingStrategy(StrEnum):
    EQUAL = "equal"
    ROOT = "root"


class FitnessEvalLimitReached(GSC):
    """
    GSC is true if the total number of fitness evaluations in the tree is greater than or equal to the limit.
    It supports different weighting strategies for the evaluations at different levels of the tree.

    The class can be initialized with a limit and an optional weighting strategy.
    The weighting strategy determines how evaluations at different levels contribute
    to the total count. Supported strategies are "equal" and "root", with "equal" being
    the default. If the strategy is "equal", all levels contribute equally to the total count.
    If the strategy is "root", only the root level contributes to the total count.

    Args:
    - limit (int): The threshold number of evaluations to check against.
    - weights (list[float] | WeightingStrategy): A list of weights corresponding to each level
    in the tree or a WeightingStrategy specifying a predefined weighting strategy. Defaults to WeightingStrategy.EQUAL.
    """

    def __init__(
        self,
        limit: int,
        weights: list[float] | WeightingStrategy | None = WeightingStrategy.EQUAL,
    ) -> None:
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
            if not isinstance(levels[i].problem, StatsGatheringProblem) and not isinstance(
                levels[i].problem, EvalCountingProblem
            ):
                raise ValueError("Problem has to be an instance of EvalCountingProblem")

            n_evals += self.weights[i] * levels[i].problem.n_evaluations

        return n_evals >= self.limit

    def _transform_weights(self, n_levels: int):
        if self.weights == WeightingStrategy.ROOT:
            self.weights = [0 for _ in range(n_levels)]
            self.weights[0] = 1
        elif self.weights == WeightingStrategy.EQUAL or self.weights is None:
            self.weights = [1 for _ in range(n_levels)]

    def __str__(self) -> str:
        return f"FitnessEvalLimitReached(limit={self.limit}, weights={self.weights})"


class SingularProblemEvalLimitReached(GSC):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def satisfied(self, tree: DemeTree) -> bool:
        problem = tree.root._problem
        if not isinstance(problem, StatsGatheringProblem) and not isinstance(problem, EvalCountingProblem):
            raise ValueError("Problem has to be an instance of EvalCountingProblem")
        return problem.n_evaluations >= self.limit

    def __str__(self) -> str:
        return f"SingularProblemEvalLimitReached(limit={self.limit})"


class SingularProblemPrecisionReached(GSC):
    def __init__(self, problem: PrecisionCutoffProblem):
        super().__init__()
        self.problem = problem

    def satisfied(self, _: DemeTree) -> bool:
        return self.problem.hit_precision

    def __str__(self) -> str:
        return f"SingularProblemPrecisionReached(precision={self.problem.precision})"


class NoActiveNonrootDemes(GSC):
    def __init__(self, n_metaepochs: int = 5) -> None:
        super().__init__()
        self.n_metaepochs = n_metaepochs

    def satisfied(self, tree: DemeTree) -> bool:
        step = tree.metaepoch_count
        for level_no in range(1, tree.height):
            if len(tree.levels[level_no]) == 0:
                return False

            for deme in tree.levels[level_no]:
                if deme.is_active or step <= deme.started_at + deme.metaepoch_count + self.n_metaepochs:
                    return False

        return True

    def __str__(self) -> str:
        return f"NoActiveNonrootDemes({self.n_metaepochs})"
