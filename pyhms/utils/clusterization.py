import numpy as np
from leap_ec.individual import Individual
from treelib import Tree

# Implementation based on A Survey of Nearest-Better Clustering in Swarm and Evolutionary Computation
class NearestBetterClustering():

    # In our particular use case we do not need weigt factor as input.
    def __init__(self, evaluated_individuals: list[Individual]) -> None:
        evaluated_individuals.sort(key=lambda ind: ind.fitness)
        self.individuals = evaluated_individuals
        self.tree = Tree()
        self.distances = []

    def cluster(self) -> list[Tree]:
        pass

    def prepare_spanning_tree(self) -> None:
        root = self.individuals[0]
        self.tree.create_node(identifier=str(root.genome), data={"genome":root.genome, "fitness":root.fitness, "distance": 0.0})
        for ind in self.individuals[1:]:
            distance, parent = self.find_nearest_better(ind, self.individuals[:self.individuals.index(ind)])
            self.tree.create_node(identifier=str(ind.genome), data={"genome":ind.genome, "fitness":ind.fitness, "distance": distance}, parent=str(parent.genome))
            self.distances.append(distance)

    def find_nearest_better(self, individual: Individual, better_individuals: list[Individual]) -> (float, Individual):
        distances = [(np.linalg.norm(individual.genome-ind.genome), ind) for ind in better_individuals]
        distances.sort(key=lambda pair: pair[0])
        return distances[0]
