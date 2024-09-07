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
        use_correction: bool | None = False,
    ) -> None:
        sorted_individuals = sorted(evaluated_individuals, reverse=True)
        self.individuals = sorted_individuals[: int(len(sorted_individuals) * truncation_factor)]
        self.tree = Tree()
        self.distance_factor = distance_factor
        self.use_correction = use_correction

    def cluster(self) -> list[Individual]:
        self._prepare_spanning_tree()
        return [node.data["individual"] for node in self._find_root_nodes()]

    @property
    def distances(self) -> list[float]:
        return [node.data["distance"] for node in self.tree.all_nodes() if not np.isinf(node.data["distance"])]

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
        correction_factor = 1 if not self.use_correction else self._get_correction_factor()
        return [
            node for node in nodes if node.data["distance"] > mean_distance * self.distance_factor * correction_factor
        ]

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

    def _get_correction_factor(self) -> float:
        """
        Calculate the correction factor cfNND, which should be applied for a random sample.
        For more details, please refer to the book:
        Preuss, M. "Multimodal optimization by means of evolutionary algorithms."
        """
        assert self.tree.size() > 0, "The tree is empty. Please run the clustering algorithm first."
        D = self.individuals[0].genome.size
        n = len(self.individuals)
        return 2.95 * (D ** (1 / 4)) * ((np.log(n) / (2 * D)) ** (1 / D))

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


class NearestBetterClusteringWithRule2(NearestBetterClustering):
    """
    Rule 2 cuts outgoing edges for individuals with at least three incoming edges
    based on the ratio of the outgoing edge length to the median of the incoming edge lengths.
    """

    def __init__(
        self,
        evaluated_individuals: list[Individual],
        distance_factor: float | None = 2.0,
        truncation_factor: float | None = 1.0,
        use_correction: bool | None = False,
        rule2_b: float = 1.0,
    ) -> None:
        super().__init__(
            evaluated_individuals=evaluated_individuals,
            distance_factor=distance_factor,
            truncation_factor=truncation_factor,
            use_correction=use_correction,
        )
        self.node_id_to_incoming_edge_lens: dict[str, list[float]] = {}
        self.rule2_b = rule2_b

    def cluster(self) -> list[Individual]:
        """
        Override the cluster method to apply Rule 2 after the spanning tree is created.
        """
        self._prepare_spanning_tree()
        self._apply_rule2_cut()
        return [node.data["individual"] for node in self._find_root_nodes()]

    def _prepare_spanning_tree(self) -> None:
        super()._prepare_spanning_tree()
        for node in self.tree.all_nodes():
            distance = node.data["distance"]
            if distance == np.inf:
                continue
            parent_id = self.tree.parent(node.identifier).identifier
            if parent_id not in self.node_id_to_incoming_edge_lens:
                self.node_id_to_incoming_edge_lens[parent_id] = []
            self.node_id_to_incoming_edge_lens[parent_id].append(distance)

    def _apply_rule2_cut(self) -> None:
        """
        Apply Rule 2 to cut outgoing edges based on the number of incoming edges.
        If the ratio of the outgoing edge to the median of incoming edges is greater than `rule2_b`,
        the outgoing edge is cut.
        """
        for node in self.tree.all_nodes():
            node_id = node.identifier
            outgoing_edge_length = node.data["distance"]
            if node_id in self.node_id_to_incoming_edge_lens and len(self.node_id_to_incoming_edge_lens[node_id]) >= 3:
                incoming_edges_lengths = self.node_id_to_incoming_edge_lens[node_id]
                median_incoming = np.median(incoming_edges_lengths)

                if outgoing_edge_length / median_incoming > self.rule2_b:
                    parent = self.tree.parent(node_id)
                    node.data["distance"] = np.inf
                    if parent:
                        self.tree.unlink_node(node_id)
