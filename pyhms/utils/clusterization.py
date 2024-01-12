import numpy as np
from leap_ec.individual import Individual
from treelib import Tree
from treelib.exceptions import DuplicatedNodeIdError

# Implementation based on A Survey of Nearest-Better Clustering in Swarm and Evolutionary Computation
class NearestBetterClustering():

    def __init__(self, evaluated_individuals: list[Individual], distance_factor: float = 2.0, truncation_factor: float = 1.0) -> None:
        evaluated_individuals.sort(key=lambda ind: ind.fitness)
        self.individuals = evaluated_individuals[:int(len(evaluated_individuals)*truncation_factor)]
        self.tree = Tree()
        self.distances = []
        self.distance_factor = distance_factor

    def cluster(self) -> list[Individual]:
        self._prepare_spanning_tree()
        nodes = self.tree.all_nodes()
        mean_distance = np.mean(self.distances)
        root_nodes = filter(lambda x: x.data["distance"] > mean_distance*self.distance_factor, nodes)
        root_individuals = [node.data["individual"] for node in root_nodes]
        return root_individuals

    def _prepare_spanning_tree(self) -> None:
        root = self.individuals[0]
        self.tree.create_node(identifier=str(root.genome), data={"individual":root, "distance": np.inf})
        for ind in self.individuals[1:]:
            # Safecheck for the case, when the individual is tied for the best fitness with root
            if ind == root:
                better_individuals = [root]
            else:
                better_individuals = self.individuals[:self.individuals.index(ind)]
            distance, parent = self._find_nearest_better(ind, better_individuals)
            try:
                self.tree.create_node(identifier=str(ind.genome), data={"individual":ind, "distance": distance}, parent=str(parent.genome))
                self.distances.append(distance)
            except DuplicatedNodeIdError:
                pass

    def _find_nearest_better(self, individual: Individual, better_individuals: list[Individual]) -> (float, Individual):
        dist_ind = [(np.linalg.norm(individual.genome-ind.genome), ind) for ind in better_individuals]
        dist_ind.sort(key=lambda pair: pair[0])
        return dist_ind[0]
