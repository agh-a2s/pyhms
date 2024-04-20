import unittest

import numpy as np
from pyhms.config import CMALevelConfig, DELevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions import DontStop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_BOUNDS, SQUARE_PROBLEM


class TestReproducibility(unittest.TestCase):
    def test_reproducibility_of_seeded_de_de_hms(self):
        options = {"random_seed": 1}
        config = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=DontStop(),
            ),
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree_1 = DemeTree(config)
        hms_tree_1.run()

        hms_tree_2 = DemeTree(config)
        hms_tree_2.run()

        best_genome_equality_mask = hms_tree_1.best_individual.genome == hms_tree_2.best_individual.genome
        self.assertEqual(np.all(best_genome_equality_mask), True, "Best individuals should be equal")
        all_genomes_1 = np.array(
            [ind.genome for _, deme in hms_tree_1.all_demes for pop in deme.history for ind in pop]
        ).flatten()
        all_genomes_2 = np.array(
            [ind.genome for _, deme in hms_tree_2.all_demes for pop in deme.history for ind in pop]
        ).flatten()
        all_genomes_equality_mask = all_genomes_1 == all_genomes_2
        self.assertEqual(np.all(all_genomes_equality_mask), True, "All genomes should be equal")

    def test_reproducibility_of_seeded_ea_cma_hms(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                sigma0=2.5,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree_1 = DemeTree(config)
        hms_tree_1.run()

        hms_tree_2 = DemeTree(config)
        hms_tree_2.run()

        best_genome_equality_mask = hms_tree_1.best_individual.genome == hms_tree_2.best_individual.genome
        self.assertEqual(np.all(best_genome_equality_mask), True, "Best individuals should be equal")
        all_genomes_1 = np.array(
            [ind.genome for _, deme in hms_tree_1.all_demes for pop in deme.history for ind in pop]
        ).flatten()
        all_genomes_2 = np.array(
            [ind.genome for _, deme in hms_tree_2.all_demes for pop in deme.history for ind in pop]
        ).flatten()
        all_genomes_equality_mask = all_genomes_1 == all_genomes_2
        self.assertEqual(np.all(all_genomes_equality_mask), True, "All genomes should be equal")
