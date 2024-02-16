import unittest

from pyhms.config import CMALevelConfig, DELevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions.usc import dont_stop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN


class TestSquare(unittest.TestCase):
    def test_square_optimization_ea(self):
        options = {"random_seed": 2}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=10,
                mutation_std=0.25,
                sample_std_dev=1.0,
                lsc=dont_stop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_cma(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_de(self):
        options = {"random_seed": 1}
        config = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")
