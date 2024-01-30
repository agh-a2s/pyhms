import unittest

import numpy as np
from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
from leap_ec.real_rep import create_real_vector
from leap_ec.representation import Representation
from pyhms.utils.clusterization import NearestBetterClustering

from .config import SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN


class TestSquare(unittest.TestCase):
    @staticmethod
    def two_squares(x) -> float:
        return min(sum(x**2), sum((x - 10.0) ** 2))

    def test_nbc_tree_calculation(self):
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=40, problem=SQUARE_PROBLEM)
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population, 2.0)
        clustering._prepare_spanning_tree()
        print(clustering.tree)

        self.assertTrue(True)

    def test_nbc_clustering_candidates(self):
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=40, problem=SQUARE_PROBLEM)
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population, 2.0)
        subtree_roots = clustering.cluster()
        print([ind.genome for ind in subtree_roots])

        self.assertTrue(True)

    def test_nbc_truncation(self):
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM))
        population = representation.create_population(pop_size=40, problem=SQUARE_PROBLEM_DOMAIN)
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population, 2.0, 0.5)
        clustering._prepare_spanning_tree()
        self.assertTrue(clustering.tree.size() == 20)

    def test_nbc_truncation_for_prepared_population(self):
        genomes_used_to_create_tree = np.array(
            [
                [0.7, 0.8],
                [1.5, 1.6],
                [0.8, 0.9],
                [1, 1],
                [1.2, 1.1],
            ]
        )
        truncated_genomes = np.array(
            [
                [200, 300],
                [500, 500],
                [700, 700],
                [800, 800],
                [1000, 1000],
            ]
        )
        population_genomes = np.concatenate([genomes_used_to_create_tree, truncated_genomes])
        population = [
            Individual(
                genome=genome,
                decoder=IdentityDecoder(),
                problem=SQUARE_PROBLEM,
            )
            for genome in population_genomes
        ]
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population, 2.0, 0.5)
        clustering._prepare_spanning_tree()

        root_id = clustering.tree.root
        root_node = clustering.tree.get_node(root_id)
        self.assertEqual(
            root_node.data["individual"],
            population[0],
            "Root should be the best individual",
        )
        self.assertEqual(
            root_node.data["distance"],
            np.inf,
            "Root distance should be infinite",
        )
        tree_nodes = clustering.tree.all_nodes()
        genomes_used_in_tree = [node.data["individual"].genome for node in tree_nodes]
        self.assertTrue(
            np.array_equal(
                np.sort(genomes_used_in_tree, axis=0),
                np.sort(genomes_used_to_create_tree, axis=0),
            ),
            "Tree should contain only genomes used to create it",
        )

        self.assertEqual(clustering.tree.size(), 5)
