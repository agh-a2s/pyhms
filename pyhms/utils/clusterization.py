import matplotlib.pyplot as plt
import numpy as np
from treelib import Node, Tree
from treelib.exceptions import DuplicatedNodeIdError

from ..core.individual import Individual
from .visualisation.dimensionality_reduction import DimensionalityReducer


def get_individual_id(individual: Individual) -> str:
    """
    Tree structure in `treelib` requires identifiers for nodes. This function returns
    a string representation of the individual's genome, which usually is unique for each individual.
    """
    return str(individual.genome)


class NearestBetterClustering:
    """
    NearestBetterClustering is a class that clusters individuals based on their fitness values.
    It uses the nearest-better clustering algorithm, which is a clustering algorithm that
    groups individuals based on their fitness values and the distance between them.

    Args:
    - evaluated_individuals: List of individuals, which have been evaluated.
    - distance_factor: A threshold multiplier. It is a predefined constant,
        that is used to scale the mean of all the distances. Default: 2.0
    - truncation_factor: The proportion of the top-performing individuals to use.
        A floating-point number between 0 and 1, where 1 would mean using
        the entire population, and a smaller value like 0.5 would mean keeping only the top 50%.
        Default: 1.0

    For more details, please refer to these papers:
    1. Luo, Wenjian, et al. "A survey of nearest-better clustering in swarm and evolutionary computation."
    2. Agrawal, Suchitra, et al. "Differential evolution with nearest better clustering for multimodal
    multiobjective optimization."
    3. Kerschke, Pascal, et al. "Detecting funnel structures by means of exploratory landscape analysis."
    """

    def __init__(
        self,
        evaluated_individuals: list[Individual],
        distance_factor: float | None = 2.0,
        truncation_factor: float | None = 1.0,
    ) -> None:
        sorted_individuals = sorted(evaluated_individuals, reverse=True)
        self.individuals = sorted_individuals[: int(len(sorted_individuals) * truncation_factor)]
        self.tree = Tree()
        self.distances: list[float] = []
        self.distance_factor = distance_factor

    def cluster(self) -> list[Individual]:
        self._prepare_spanning_tree()
        return [node.data["individual"] for node in self._find_root_nodes()]

    def _prepare_spanning_tree(self) -> None:
        root = self.individuals[0]
        self.tree.create_node(
            identifier=get_individual_id(root),
            data={"individual": root, "distance": np.inf},
        )

        for ind in self.individuals[1:]:
            # Safecheck for the case, when the individual is tied for the best fitness with root
            if ind == root:
                better_individuals = [root]
            else:
                better_individuals = self.individuals[: self.individuals.index(ind)]
            distance, parent = self._find_nearest_better(ind, better_individuals)
            try:
                self.tree.create_node(
                    identifier=get_individual_id(ind),
                    data={"individual": ind, "distance": distance},
                    parent=get_individual_id(parent),
                )
                self.distances.append(distance)
            except DuplicatedNodeIdError:
                pass

    def _find_root_nodes(self) -> list[Node]:
        nodes = self.tree.all_nodes()
        mean_distance = np.mean(self.distances)
        return [node for node in nodes if node.data["distance"] > mean_distance * self.distance_factor]

    def _find_nearest_better(
        self, individual: Individual, better_individuals: list[Individual]
    ) -> tuple[float, Individual]:
        better_genomes = np.array([ind.genome for ind in better_individuals])
        distances = np.linalg.norm(individual.genome - better_genomes, axis=1)
        nearest_better_index = np.argmin(distances)
        return distances[nearest_better_index], better_individuals[nearest_better_index]

    def _find_cluster_center(self, node: Node, root_node_ids: list[str]) -> Node | None:
        while node and node.identifier not in root_node_ids:
            node = self.tree.parent(node.identifier)
        return node if node else None

    def assign_clusters(self) -> dict:
        """
        Assign each individual to a cluster.
        Returns a dictionary where keys are cluster centers and values are lists of individuals in each cluster.
        """
        root_node_ids = [node.identifier for node in self._find_root_nodes()]
        clusters: dict[str, list[Individual]] = {}
        for node in self.tree.all_nodes():
            cluster_center = self._find_cluster_center(node, root_node_ids)
            if not cluster_center:
                continue
            cluster_center_id = get_individual_id(cluster_center.data["individual"])
            if cluster_center_id not in clusters:
                clusters[cluster_center_id] = [cluster_center.data["individual"]]
            clusters[cluster_center_id].append(node.data["individual"])
        return clusters

    def plot_clusters(self, dimensionality_reducer: DimensionalityReducer) -> None:
        """
        Plot the clustered individuals.
        """
        clusters = self.assign_clusters()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
        plt.figure(figsize=(10, 8))

        dimensionality_reducer.fit(np.array([ind.genome for ind in self.individuals]))

        for color, (cluster_center_id, members) in zip(colors, clusters.items()):
            members_genomes = np.array([ind.genome for ind in members])
            cluster_center = self.tree.get_node(cluster_center_id).data["individual"].genome
            reduced_members_genomes = dimensionality_reducer.transform(members_genomes)
            plt.scatter(
                reduced_members_genomes[:, 0],
                reduced_members_genomes[:, 1],
                color=color,
                label=f"Cluster centered at {cluster_center_id}",
            )
            reduced_cluster_center = dimensionality_reducer.transform(np.array([cluster_center]))
            plt.scatter(
                reduced_cluster_center[:, 0],
                reduced_cluster_center[:, 1],
                color=color,
                marker="x",
            )
        plt.title("Nearest Better Clustering")
        plt.xlabel("Reduced Dimension 1")
        plt.ylabel("Reduced Dimension 2")
        if len(clusters) < 5:
            plt.legend()
        plt.show()
