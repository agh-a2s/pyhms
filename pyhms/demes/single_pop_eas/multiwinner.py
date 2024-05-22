from typing import Callable, Protocol

import numpy as np

from ...core.population import Population
from ...core.problem import Problem
from .common import VariationalOperator


class UtilityFunction:
    """
    Utility function for multiwinner voting.

    For more details, please refer to these papers:
    1. Faliszewski, Piotr, et al. "Multiwinner voting in genetic algorithms for
    solving ill-posed global optimization problems."
    """

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
        self.h = (lambda x: x) if problem.maximize else (lambda x: 1 / (1 + x))
        reversed_fitness = self.h(candidate_fitness)
        distance_value = self.distance(voter_genome, candidate_genome)
        return self.gamma(reversed_fitness) * self.delta(distance_value)

    def evaluate_population(self, population: Population) -> np.ndarray:
        # Multiwinner behavior depends on the sign of the fitness values.
        # In order to use 1/(1+x) as a reversal function, fitness values must be different than -1.
        positive_fitnesses = population.fitnesses - np.min(population.fitnesses)
        scores = []
        for voter_genome in population.genomes:
            candidate_scores = []
            for candidate_genome, candidate_fitness in zip(population.genomes, positive_fitnesses):
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
    """
    i-th index of the j-th row of the output array is the position of the i-th candidate in the j-th preference list.
    """
    assert preferences.ndim == 2, "Input must be a 2D array"
    assert preferences.shape[0] == preferences.shape[1], "Input must be a square matrix"
    N = preferences.shape[0]
    positions_in_sorted_array = np.empty_like(preferences)
    row_indices = np.arange(N)[:, None]
    positions_in_sorted_array[row_indices, preferences] = np.tile(np.arange(N), (N, 1))
    return positions_in_sorted_array


class MWPolicy(Protocol):
    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        ...


class SNTVPolicy(MWPolicy):
    """
    Under SNTV we pick the k candidates with the highest plurality scores
    (i.e., k candidates ranked first most frequently).
    """

    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        best_candidate_indices = preferences[:, 0].flatten()
        unique_candidate_indices, counts = np.unique(best_candidate_indices, return_counts=True)
        return unique_candidate_indices[np.argsort(-counts)[:k]]


class BordaPolicy(MWPolicy):
    """
    Under k-Borda we pick k candidates with the highest Borda scores.
    Borda score is defined as follows (N is the total number of candidates,
    i is the position of the candidate in the preference list).
    .. math::

    \beta(i) = N - i
    """

    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        N = preferences.shape[1]
        positions_in_sorted_array = get_positions_in_preferences(preferences)
        borda_scores = (N - positions_in_sorted_array).sum(axis=0)
        return np.argsort(-borda_scores)[:k]


class BlocPolicy(MWPolicy):
    """
    Under Bloc we pick the k candidates with the highest k-Approval scores.
    """

    def __call__(self, preferences: np.ndarray, k: int) -> np.ndarray:
        top_k = preferences[:, :k].flatten()
        unique_candidate_indices, counts = np.unique(top_k, return_counts=True)
        return unique_candidate_indices[np.argsort(-counts)[:k]]


class MultiwinnerSelection(VariationalOperator):
    def __init__(
        self,
        utility_function: UtilityFunction,
        voting_scheme: MWPolicy,
        k: int,
    ) -> None:
        self.utility_function = utility_function
        self.voting_scheme = voting_scheme
        self.k = k

    def __call__(self, population: Population) -> Population:
        preferences = self.utility_function.get_preferences(population)
        selected_indices = self.voting_scheme(preferences, self.k)
        return population[selected_indices]
