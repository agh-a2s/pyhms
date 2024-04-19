import unittest

from pyhms.config import CMALevelConfig, DELevelConfig, EALevelConfig, LocalOptimizationConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import (
    DemeLimit,
    LevelLimit,
    NBC_FarEnough,
    NBCGeneratorWithLocalMethod,
    SkipSameSprout,
    SproutMechanism,
)
from pyhms.stop_conditions import DontStop
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
                lsc=DontStop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=10,
                mutation_std=0.25,
                sample_std_dev=1.0,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_cma_warm_start(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=None,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_cma_warm_start_set_stds(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=None,
                set_stds=True,
                lsc=DontStop(),
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
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_with_local_method(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=DontStop(),
            ),
            LocalOptimizationConfig(
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                lsc=DontStop(),
                maxiter=10,
            ),
        ]

        sprout_cond = SproutMechanism(
            NBCGeneratorWithLocalMethod(3.0, 0.7),
            [NBC_FarEnough(3.0, 2), DemeLimit(1)],
            [LevelLimit(4), SkipSameSprout()],
        )

        config = TreeConfig(config, DEFAULT_GSC, sprout_cond, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 3, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")
