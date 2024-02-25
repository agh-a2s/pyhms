from abc import ABC, abstractmethod
from enum import Enum

from ..problem import EvalCountingProblem, PrecisionCutoffProblem, StatsGatheringProblem
from ..tree import DemeTree


class GlobalStopCondition(ABC):
    @abstractmethod
    def __call__(self, tree: DemeTree) -> bool:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__


class RootStopped(GlobalStopCondition):
    """
    GSC is true if the root is not active.
    """

    def __call__(self, tree: DemeTree) -> bool:
        return not tree.root.is_active


class AllStopped(GlobalStopCondition):
    """
    GSC is true if there are no active demes in the tree.
    """

    def __call__(self, tree: DemeTree) -> bool:
        return len(list(tree.active_demes)) == 0


class WeightingStrategy(str, Enum):
    EQUAL = "equal"
    ROOT = "root"


class FitnessEvalLimitReached(GlobalStopCondition):
    """
    GSC is true if the total number of fitness evaluations in the tree is greater than or equal to the limit.
    It supports different weighting strategies for the evaluations at different levels of the tree.
    It should be used if different levels of the tree use different problems,
    otherwise use SingularProblemEvalLimitReached.

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
        self.limit = limit
        self.weights = weights

    def __call__(self, tree: DemeTree) -> bool:
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


class SingularProblemEvalLimitReached(GlobalStopCondition):
    """
    GSC is true if the total number of fitness evaluations in the tree is greater than or equal to the limit.
    It assumes that the same problem is used at all levels of the tree.

    Args:
    - limit (int): The threshold number of evaluations to check against.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def __call__(self, tree: DemeTree) -> bool:
        problem = tree.root._problem
        if not isinstance(problem, StatsGatheringProblem) and not isinstance(problem, EvalCountingProblem):
            raise ValueError("Problem has to be an instance of EvalCountingProblem")
        return problem.n_evaluations >= self.limit

    def __str__(self) -> str:
        return f"SingularProblemEvalLimitReached(limit={self.limit})"


class SingularProblemPrecisionReached(GlobalStopCondition):
    """
    GSC is true if the precision of the problem is reached.

    Args:
    - problem (PrecisionCutoffProblem): The problem to check the precision for.
    """

    def __init__(self, problem: PrecisionCutoffProblem):
        self.problem = problem

    def __call__(self, tree: DemeTree) -> bool:
        return self.problem.hit_precision

    def __str__(self) -> str:
        return f"SingularProblemPrecisionReached(precision={self.problem.precision})"


class NoActiveNonrootDemes(GlobalStopCondition):
    """
    GSC is true if there are no active non-root demes in the tree for a certain number of metaepochs.

    Args:
    - n_metaepochs (int): The number of metaepochs to wait before the condition is satisfied. Default: 5.
    """

    def __init__(self, n_metaepochs: int = 5) -> None:
        self.n_metaepochs = n_metaepochs

    def __call__(self, tree: DemeTree) -> bool:
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
