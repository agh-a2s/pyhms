import unittest

from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions import DontRun, DontStop, MetaepochLimit, UniversalStopCondition
from pyhms.tree import DemeTree

from .config import DEFAULT_SPROUT_COND, SQUARE_BOUNDS, SQUARE_PROBLEM

POPULATION_SIZE = 50


class TestUniversalStopCondition(unittest.TestCase):
    def get_deme_tree(self, gsc: UniversalStopCondition) -> DemeTree:
        levels = [
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
        tree_config = TreeConfig(
            levels=levels,
            gsc=gsc,
            sprout_mechanism=DEFAULT_SPROUT_COND,
            options={},
        )

        deme_tree = DemeTree(tree_config)
        return deme_tree

    def test_metaepoch_limit_for_deme_tree(self):
        usc = MetaepochLimit(2)
        deme_tree = self.get_deme_tree(usc)
        self.assertFalse(usc(deme_tree))
        deme_tree.metaepoch_count = 2
        self.assertTrue(usc(deme_tree))

    def test_dont_stop(self):
        usc = DontStop()
        deme_tree = self.get_deme_tree(usc)
        self.assertFalse(usc(deme_tree))

    def test_dont_run(self):
        usc = DontRun()
        deme_tree = self.get_deme_tree(usc)
        self.assertTrue(usc(deme_tree))
