from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from typing import Tuple
from .core.problem import Problem

class MergeCondition(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def can_merge(self, cluster1: np.ndarray, cluster2: np.ndarray) -> bool:
        pass

    @staticmethod
    def find_closest_points(cluster1: np.ndarray, cluster2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        distances = cdist(cluster1, cluster2)
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        return cluster1[i], cluster2[j]

class LocalOptimizationMergeCondition(MergeCondition):
    def __init__(self, problem: Problem, epsilon: float = 1e-6):
        super().__init__(problem)
        self.epsilon = epsilon

    def can_merge(self, cluster1: np.ndarray, cluster2: np.ndarray) -> bool:
        p1, p2 = self.find_closest_points(cluster1, cluster2)
        
        local_p1 = self._local_optimization(p1)
        local_p2 = self._local_optimization(p2)
        
        return self._is_in_extension(local_p1, cluster2) or self._is_in_extension(local_p2, cluster1)

    def _local_optimization(self, start_point: np.ndarray) -> np.ndarray:
        objective = lambda x: (-1 if self.problem.maximize else 1) * self.problem.evaluate(x)
        result = minimize(fun=objective, x0=start_point, method='L-BFGS-B', bounds=self.problem.bounds)
        return result.x

    def _is_in_extension(self, point: np.ndarray, cluster: np.ndarray) -> bool:
        distances = cdist([point], cluster)
        return np.min(distances) <= self.epsilon

class HillValleyMergeCondition(MergeCondition):
    def __init__(self, problem: Problem, k: int):
        super().__init__(problem)
        self.k = k

    def can_merge(self, cluster1: np.ndarray, cluster2: np.ndarray) -> bool:
        p1, p2 = self.find_closest_points(cluster1, cluster2)
        return self._hill_valley_function(p1, p2) == 0

    def _hill_valley_function(self, p1: np.ndarray, p2: np.ndarray) -> float:
        f1, f2 = self.problem.evaluate(p1), self.problem.evaluate(p2)
        max_fitness = max(f1, f2) if self.problem.maximize else min(f1, f2)
        
        for j in range(1, self.k + 1):
            t = j / (self.k + 1)
            inter_j = p1 + t * (p2 - p1)
            fitness_inter_j = self.problem.evaluate(inter_j)
            
            if self.problem.worse_than(max_fitness, fitness_inter_j):
                return abs(fitness_inter_j - max_fitness)
        
        return 0
