import unittest

from pyhms.config import DELevelConfig, CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions.usc import dont_stop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, DEFAULT_NBC_SPROUT_COND, SQUARE_PROBLEM, SQUARE_BOUNDS, LEVEL_LIMIT

class Test3Levels(unittest.TestCase):
    def test_with_simple_sprout(self):
        options = {"log_level": "debug"}
        levels = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=dont_stop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]
        tree_config = TreeConfig(
            levels=levels,
            gsc=DEFAULT_GSC,
            sprout_mechanism=DEFAULT_SPROUT_COND,
            options=options,
        )

        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        for level in deme_tree.levels:
            self.assertTrue(len([deme for deme in level if deme.is_active]) <= LEVEL_LIMIT)


    def test_with_NBC_sprout(self):
        options = {"log_level": "debug"}
        levels = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=dont_stop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_BOUNDS,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]
        tree_config = TreeConfig(
            levels=levels,
            gsc=DEFAULT_GSC,
            sprout_mechanism=DEFAULT_SPROUT_COND,
            options=options,
        )

        deme_tree = DemeTree(tree_config)
        deme_tree.run()
        for level in deme_tree.levels:
            self.assertTrue(len([deme for deme in level if deme.is_active]) <= LEVEL_LIMIT)