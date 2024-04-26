import unittest

from pyhms import minimize
from pyhms.config import (
    CMALevelConfig,
    DELevelConfig,
    EALevelConfig,
    LHSLevelConfig,
    LocalOptimizationConfig,
    TreeConfig,
)
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

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_BOUNDS, SQUARE_PROBLEM, square


class TestSquare(unittest.TestCase):
    def test_square_optimization_ea(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=4,
                problem=SQUARE_PROBLEM,
                pop_size=20,
                mutation_std=0.25,
                sample_std_dev=0.5,
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
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
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
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
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
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                sigma0=2.5,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_lhs(self):
        options = {"random_seed": 1}
        config = [
            LHSLevelConfig(
                problem=SQUARE_PROBLEM,
                pop_size=20,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                sigma0=2.5,
                lsc=DontStop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertEqual(hms_tree.height, 2, "Tree height should be equal 2")
        self.assertLessEqual(hms_tree.best_individual.fitness, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_scipy_style(self):
        max_iter = 5
        result = minimize(square, SQUARE_BOUNDS, log_level="DEBUG", maxiter=max_iter)
        self.assertEqual(result.nit, max_iter)
        self.assertLessEqual(result.fun, 1e-3, "Best fitness should be close to 0")

    def test_square_optimization_with_local_method(self):
        options = {"random_seed": 1}
        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                sigma0=2.5,
                lsc=DontStop(),
            ),
            LocalOptimizationConfig(
                problem=SQUARE_PROBLEM,
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
