import unittest

import numpy as np
from pyhms.core.individual import Individual
from pyhms.utils.samplers import sample_uniform
from pyhms.utils.clusterization import NearestBetterClustering, get_individual_id

from .config import NEGATIVE_SQUARE_PROBLEM, SQUARE_BOUNDS, SQUARE_PROBLEM


class TestClustering(unittest.TestCase):
    def test_nbc_truncation(self):
        truncation_factor = 0.3
        population_size = 40
        population = Individual.create_population(
            pop_size=population_size,
            problem=SQUARE_PROBLEM,
            initialize=sample_uniform(bounds=SQUARE_BOUNDS),
        )
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population, truncation_factor=truncation_factor)
        clustering._prepare_spanning_tree()
        self.assertEqual(clustering.tree.size(), int(len(population) * truncation_factor))

    def test_nbc_with_truncation_for_prepared_population(self):
        genomes_used_to_create_tree = np.array(
            [
                [0.7, 0.8],
                [1.5, 1.6],
                [0.8, 0.9],
                [1, 1],
                [1.2, 1.1],
                [-0.8, -0.9],
            ]
        )
        truncated_genomes = np.array([[200, 300], [500, 500], [700, 700], [800, 800], [1000, 1000], [2000, 2000]])
        population_genomes = np.concatenate([genomes_used_to_create_tree, truncated_genomes])
        population = [
            Individual(
                genome=genome,
                problem=SQUARE_PROBLEM,
            )
            for genome in population_genomes
        ]
        Individual.evaluate_population(population)
        population_copy = population.copy()

        clustering = NearestBetterClustering(population, truncation_factor=0.5)
        clustering._prepare_spanning_tree()

        root_node = clustering.tree.get_node(clustering.tree.root)
        tree_nodes = clustering.tree.all_nodes()
        genomes_used_in_tree = [node.data["individual"].genome for node in tree_nodes]

        self.assertTrue(
            np.array_equal(
                np.sort(genomes_used_in_tree, axis=0),
                np.sort(genomes_used_to_create_tree, axis=0),
            ),
            "Tree should contain only genomes used to create it",
        )

        # Check tree structure, it should look like this:
        # [0.7 0.8]
        # ├── [-0.8 -0.9]
        # └── [0.8 0.9]
        #     └── [1. 1.]
        #         └── [1.2 1.1]
        #             └── [1.5 1.6]
        self.assertEqual(
            root_node.data["individual"],
            population_copy[0],
            "Root should be the best individual",
        )
        self.assertEqual(
            root_node.data["distance"],
            np.inf,
            "Root distance should be infinite",
        )
        # I use idx from `genomes_used_to_create_tree`.
        child_idx_to_parent_idx = {2: 0, 3: 2, 4: 3, 1: 4, 5: 0}
        for child_idx, parent_idx in child_idx_to_parent_idx.items():
            child_identifier = get_individual_id(population_copy[child_idx])
            parent_from_tree = clustering.tree.parent(child_identifier)
            self.assertEqual(
                parent_from_tree.data["individual"],
                population_copy[parent_idx],
                f"{parent_from_tree.identifier} should be the parent of {child_identifier}",
            )
            self.assertAlmostEqual(
                clustering.tree.get_node(child_identifier).data["distance"],
                np.linalg.norm(population_copy[child_idx].genome - population_copy[parent_idx].genome),
            )
        self.assertEqual(len(tree_nodes), len(genomes_used_in_tree))

    def test_nbc_works_for_min_and_max(self):
        population_size = 40
        min_population = Individual.create_population(
            pop_size=population_size,
            problem=SQUARE_PROBLEM,
            initialize=sample_uniform(bounds=SQUARE_BOUNDS),
        )
        max_population = [
            Individual(
                genome=ind.genome,
                problem=NEGATIVE_SQUARE_PROBLEM,
            )
            for ind in min_population
        ]
        Individual.evaluate_population(min_population)
        Individual.evaluate_population(max_population)

        min_clustering = NearestBetterClustering(min_population)
        min_root_nodes = min_clustering.cluster()
        max_clustering = NearestBetterClustering(max_population)
        max_root_nodes = max_clustering.cluster()

        self.assertEqual(
            [ind.genome for ind in min_root_nodes],
            [ind.genome for ind in max_root_nodes],
        )
