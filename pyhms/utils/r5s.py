import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.individual import Individual
from ..core.problem import get_function_problem


class R5SSelection:
    """
    For more information, see Preuss, M., and Wessing. S.
    "Measuring multimodal optimization solution sets with a view to multiobjective techniques."
    """

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def __call__(self, individuals: list[Individual], n: int = 5) -> list[Individual]:
        if len(individuals) <= n:
            return individuals
        minimize = not get_function_problem(individuals[0].problem).maximize
        sorted_individuals = sorted(individuals, reverse=minimize)
        distances = self._get_distances(sorted_individuals)
        nearest_distances = self._get_nearest_distances(distances)
        nearest_better_distances = self._get_nearest_better_distances(distances)
        weighted_worse_distances = self._get_total_weighted_worse_distances(distances)
        interesting_individuals, weighted_worse_distances = self._filter_uninteresting_points(
            sorted_individuals,
            nearest_distances,
            nearest_better_distances,
            weighted_worse_distances,
        )
        selected_individuals = interesting_individuals[: self.top_k]
        for idx in range(len(interesting_individuals) - 1, self.top_k - 1, -1):
            # Check if better point with higher ds(point) exists:
            if (weighted_worse_distances[:idx] > weighted_worse_distances[idx]).any():
                selected_individuals.append(interesting_individuals[idx])
        return selected_individuals

    def _get_distances(self, individuals: list[Individual]) -> np.ndarray:
        distances = pairwise_distances([individual.genome for individual in individuals])
        np.fill_diagonal(distances, np.inf)
        return distances

    def _get_nearest_better_distances(self, distances: np.ndarray) -> np.ndarray:
        nearest_better_distances = []
        for idx in range(len(distances)):
            if idx == 0:
                nearest_better_distances.append(0)
            else:
                nearest_better_distances.append(np.min(distances[idx, :idx]))
        return np.array(nearest_better_distances)

    def _get_nearest_distances(self, distances: np.ndarray) -> np.ndarray:
        return np.min(distances, axis=1)

    def _get_total_weighted_worse_distances(self, distances: np.ndarray) -> np.ndarray:
        weighted_worse_distances = []
        for idx in range(len(distances)):
            if idx == len(distances) - 1:
                weighted_worse_distances.append(0)
            else:
                worse_individual_distances = distances[idx, (idx + 1) :]
                weights = 1 / 2 ** np.arange(1, len(worse_individual_distances) + 1)
                weighted_worse_distances.append(np.sum(worse_individual_distances * weights))
        return np.array(weighted_worse_distances)

    def _filter_uninteresting_points(
        self,
        individuals: list[Individual],
        nearest_distances: np.ndarray,
        nearest_better_distances: np.ndarray,
        weighted_worse_distances: np.ndarray,
    ) -> tuple[list[Individual], np.ndarray]:
        interesting_points = np.logical_not(np.isclose(nearest_distances, nearest_better_distances))
        interesting_individuals = [
            individuals[idx] for idx, is_interesting in enumerate(interesting_points) if is_interesting
        ]
        return interesting_individuals, weighted_worse_distances[interesting_points]
