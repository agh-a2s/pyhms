import numpy as np
from sklearn.metrics import pairwise_distances

from ..core.individual import Individual


class R5SSelection:
    """
    For more information, see:
    Preuss, M., and Wessing. S. "Measuring multimodal optimization solution sets with a view to multiobjective techniques." # noqa: E501
    """

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def __call__(self, individuals: list[Individual], n: int = 5) -> list[Individual]:
        if len(individuals) <= n:
            return individuals
        sorted_individuals = sorted(individuals, reverse=individuals[0].problem.maximize)
        distances = pairwise_distances([individual.genome for individual in sorted_individuals])
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        nearest_better_distances = []
        for idx in range(len(sorted_individuals)):
            if idx == 0:
                nearest_better_distances.append(0)
            else:
                nearest_better_distances.append(np.min(distances[idx, :idx]))
        weighted_worse_distances = []
        for idx in range(len(sorted_individuals)):
            if idx == len(sorted_individuals) - 1:
                weighted_worse_distances.append(0)
            else:
                worse_individual_distances = distances[idx, (idx + 1) :]
                weights = 1 / 2 ** np.arange(1, len(worse_individual_distances) + 1)
                weighted_worse_distances.append(np.sum(worse_individual_distances * weights))
        # Instead of removing, we keep the selected individuals:
        selected_individuals = sorted_individuals[: self.top_k]
        # Iterate over individuals in reverse order:
        for idx in range(len(sorted_individuals) - 1, self.top_k - 1, -1):
            # Always skip uninteresting individuals:
            if min_distances[idx] == nearest_better_distances[idx]:
                continue
            # Check if better point with higher ds(point) exists:
            weighted_worse_distance = weighted_worse_distances[idx]
            if (weighted_worse_distances[:idx] > weighted_worse_distance).any():
                selected_individuals.append(sorted_individuals[idx])
        return selected_individuals
