from typing import Callable, Protocol

import numpy as np

from ...core.population import Population
from ...core.problem import Problem


class UtilityFunction:
    def __init__(self, distance: Callable, gamma: Callable, delta: Callable) -> None:
        self.distance = distance
        self.gamma = gamma
        self.delta = delta

    def evaluate(
        self,
        voter_genome: np.ndarray,
        candidate_genome: np.ndarray,
        candidate_fitness: float,
        problem: Problem,
    ) -> float:
        self.h = lambda x: x if problem.maximize else lambda x: -1 * x
        return self.gamma(self.h(candidate_fitness)) * self.delta(self.distance(voter_genome, candidate_genome))

    def evaluate_population(self, population: Population) -> np.ndarray:
        scores = []
        for voter_genome in population.genomes:
            candidate_scores = []
            for candidate_genome, candidate_fitness in zip(population.genomes, population.fitnesses):
                if np.array_equal(voter_genome, candidate_genome):
                    candidate_scores.append(-np.inf)
                else:
                    candidate_scores.append(
                        self.evaluate(
                            voter_genome=voter_genome,
                            candidate_genome=candidate_genome,
                            candidate_fitness=candidate_fitness,
                            problem=population.problem,
                        )
                    )
            scores.append(candidate_scores)
        return np.array(scores)

    def get_preferences(self, population: Population) -> np.ndarray:
        scores = self.evaluate_population(population)
        return np.argsort(-1 * scores)


def get_positions_in_preferences(preferences: np.ndarray) -> np.ndarray:
    assert preferences.ndim == 2, "Input must be a 2D array"
    assert preferences.shape[0] == preferences.shape[1], "Input must be a square matrix"
    N = preferences.shape[0]
    positions_in_sorted_array = np.empty_like(preferences)
    row_indices = np.arange(N)[:, None]
    positions_in_sorted_array[row_indices, preferences] = np.tile(np.arange(N), (N, 1))
    return positions_in_sorted_array


class MultiwinnerVoting(Protocol):
    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        ...


class SNTVoting(MultiwinnerVoting):
    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        best_candidate_indices = preferences[:, 0].flatten()
        unique_candidate_indices, counts = np.unique(best_candidate_indices, return_counts=True)
        return unique_candidate_indices[np.argsort(-counts)[:k]]


class BordaVoting(MultiwinnerVoting):
    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        N = preferences.shape[1]
        positions_in_sorted_array = get_positions_in_preferences(preferences)
        borda_scores = (N - positions_in_sorted_array).sum(axis=0)
        return np.argsort(-borda_scores)[:k]


class BlocVoting(MultiwinnerVoting):
    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        top_k = preferences[:, :k].flatten()
        unique_candidate_indices, counts = np.unique(top_k, return_counts=True)
        return unique_candidate_indices[np.argsort(-counts)[:k]]
