import numpy as np

from .cluster import Cluster
from .merge_conditions import MergeCondition


class PairwiseNeighborConsolidator:
    def __init__(
        self,
        merge_condition: MergeCondition,
        max_distance: float | None = None,
    ):
        self.max_distance = max_distance
        self.merge_condition = merge_condition

    def reduce_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        first_cluster_idx = 0
        second_cluster_idx = 0

        while True:
            first_cluster_idx, second_cluster_idx = self.find_neighbors(clusters, first_cluster_idx, second_cluster_idx)
            if first_cluster_idx is None and second_cluster_idx is None:
                break

            first_cluster, second_cluster = (
                clusters[first_cluster_idx],
                clusters[second_cluster_idx],
            )
            if self.merge_condition.can_merge(first_cluster, second_cluster):
                merged = Cluster.merge([first_cluster, second_cluster])
                clusters = [
                    cluster
                    for idx, cluster in enumerate(clusters)
                    if idx not in (first_cluster_idx, second_cluster_idx)
                ]
                clusters.append(merged)
                first_cluster_idx, second_cluster_idx = 0, 0

        return clusters

    def find_neighbors(
        self,
        clusters: list[Cluster],
        first_cluster_idx: int,
        second_cluster_idx: int,
    ) -> tuple[int | None, int | None]:
        for i in range(first_cluster_idx, len(clusters) - 1):
            for j in range(max(i + 1, second_cluster_idx + 1), len(clusters)):
                if i == first_cluster_idx and j <= second_cluster_idx:
                    continue

                distance = np.linalg.norm(clusters[i].mean - clusters[j].mean)
                if self.max_distance is not None and distance < self.max_distance:
                    return i, j
        return None, None
