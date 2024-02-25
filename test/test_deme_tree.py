import unittest

from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.cma_deme import CMADeme
from pyhms.demes.ea_deme import EADeme
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions.usc import DontStop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN


class TestDemeTree(unittest.TestCase):
    def test_deme_tree_properties(self):
        levels = [
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
        ]
        config = TreeConfig(levels, DEFAULT_GSC, DEFAULT_SPROUT_COND, options={})
        hms_tree = DemeTree(config)
        hms_tree.run()
        self.assertIsInstance(hms_tree.root, EADeme)
        self.assertEqual(hms_tree.height, len(levels))
        for leave in hms_tree.leaves:
            self.assertIsInstance(leave, CMADeme)
        self.assertGreaterEqual(hms_tree.best_individual, hms_tree.best_individual)
        for level, deme in hms_tree.all_demes:
            self.assertLessEqual(level, hms_tree.height)
            self.assertLessEqual(deme.best_individual, hms_tree.best_individual)
            self.assertLessEqual(deme.best_current_individual, deme.best_individual)
            self.assertGreaterEqual(deme.n_evaluations, 0)
