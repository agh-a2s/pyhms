"""
    Global stopping conditions.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Union

from .problem import StatsGatheringProblem
from .tree import DemeTree

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
            if not isinstance(levels[i].problem, StatsGatheringProblem):
                raise ValueError("Problem has to be an instance of StatsGatheringProblem")

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
