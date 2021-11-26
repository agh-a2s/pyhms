"""
    Sprouting conditions.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy.linalg as nla

from .deme import Deme
from .tree import DemeTree

class sprout_condition(ABC):
    @abstractmethod
    def sprout_possible(self, deme: Deme, level: int, tree: DemeTree) -> bool:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sprout_possible(*args, **kwds)

class level_limit(sprout_condition):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def sprout_possible(self, deme: Deme, level: int, tree: DemeTree) -> bool:
        return len(tree.level(level)) < self.limit

    def __str__(self) -> str:
        return f"level_limit({self.limit})"

class far_enough(sprout_condition):
    def __init__(self, min_distance: Union[List[float], float], norm_ord: int = 2) -> None:
        super().__init__()
        if not isinstance(min_distance, list) and not isinstance(min_distance, float):
            raise ValueError("min_distance must be a float or a list of floats")

        self.min_distance = min_distance
        self.norm_ord = norm_ord

    def sprout_possible(self, deme: Deme, level: int, tree: DemeTree) -> bool:
        child_siblings = tree.level(level + 1)
        child_seed = max(deme.population)
        if isinstance(self.min_distance, list):
            min_dist = self.min_distance[level]
        elif isinstance(self.min_distance, float):
            min_dist = self.min_distance

        for sibling in child_siblings:
            if nla.norm(child_seed.genome - sibling.centroid, ord=self.norm_ord) <= min_dist:
                return False
        return True

    def __str__(self) -> str:
        par_str = f"dist={self.min_distance}"
        if self.norm_ord != 2:
            par_str += f", ord={self.norm_ord}"
        return f"far_enough({par_str})"
