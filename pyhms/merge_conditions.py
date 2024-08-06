from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .core.population import Population
from .core.problem import Problem
from .demes.cma_deme import CMADeme


def mahalanobis_distance(x: np.ndarray, y: np.ndarray, covariance_matrix_inverse: np.ndarray) -> float:
    diff = x - y
    return np.sqrt(diff @ covariance_matrix_inverse @ diff)


class Cluster:
    def __init__(
        self,
        population: Population,
        mean: np.ndarray,
        covariance_matrix: np.ndarray,
    ):
        self.population = population
        self.mean = mean
        self.covariance_matrix_inverse = np.linalg.inv(covariance_matrix)

    @classmethod
    def from_deme(cls, deme: CMADeme) -> "Cluster":
        return cls(
            Population.from_individuals(deme.current_population),
            deme.mean,
            deme.covariance_matrix,
        )

    def is_in_extension(self, point: np.ndarray, threshold: float = 1) -> bool:
        return mahalanobis_distance(point, self.mean, self.covariance_matrix_inverse) < threshold


class MergeCondition(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def can_merge(self, cluster1: Cluster, cluster2: Cluster) -> bool:
        pass

    @staticmethod
    def find_closest_points(cluster1: Cluster, cluster2: Cluster) -> tuple[np.ndarray, np.ndarray]:
        distances = cdist(cluster1.population.genomes, cluster2.population.genomes)
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        return cluster1.population.genomes[i], cluster2.population.genomes[j]


class LocalOptimizationMergeCondition(MergeCondition):
    def __init__(self, problem: Problem, epsilon: float = 1e-6):
        super().__init__(problem)
        self.epsilon = epsilon

    def can_merge(self, cluster1: Cluster, cluster2: Cluster) -> bool:
        p1, p2 = self.find_closest_points(cluster1, cluster2)

        local_p1 = self._local_optimization(p1)
        local_p2 = self._local_optimization(p2)

        return cluster2.is_in_extension(local_p1) or cluster1.is_in_extension(local_p2)

    def _local_optimization(self, start_point: np.ndarray) -> np.ndarray:
        def objective(x):
            return (-1 if self.problem.maximize else 1) * self.problem.evaluate(x)

        result = minimize(fun=objective, x0=start_point, method="L-BFGS-B", bounds=self.problem.bounds)
        return result.x


class HillValleyMergeCondition(MergeCondition):
    def __init__(self, problem: Problem, k: int):
        super().__init__(problem)
        self.k = k

    def can_merge(self, cluster1: Cluster, cluster2: Cluster) -> bool:
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
