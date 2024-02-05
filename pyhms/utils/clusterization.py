import numpy as np
from leap_ec.individual import Individual
from treelib import Tree
from treelib.exceptions import DuplicatedNodeIdError


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

    Implementation based on a following paper:
    Luo, Wenjian & Lin, Xin & Zhang, Jiajia & Preuss, Mike. (2021).
    A Survey of Nearest-Better Clustering in Swarm and Evolutionary Computation. 10.1109/CEC45853.2021.9505008.
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
        nodes = self.tree.all_nodes()
        mean_distance = np.mean(self.distances)
        root_nodes = filter(lambda x: x.data["distance"] > mean_distance * self.distance_factor, nodes)
        root_individuals = [node.data["individual"] for node in root_nodes]
        return root_individuals

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

    def _find_nearest_better(
        self, individual: Individual, better_individuals: list[Individual]
    ) -> tuple[float, Individual]:
        better_genomes = np.array([ind.genome for ind in better_individuals])
        distances = np.linalg.norm(individual.genome - better_genomes, axis=1)
        nearest_better_index = np.argmin(distances)
        return distances[nearest_better_index], better_individuals[nearest_better_index]
