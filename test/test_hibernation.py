import unittest

from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.new_sea import SEA
from pyhms.stop_conditions import DontStop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM


class TestHibernation(unittest.TestCase):
    def test_deme_tree(self):
        options = {"hibernation": True, "log_level": "debug"}
        levels = [
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
        ]
        tree_config = TreeConfig(
            levels=levels,
            gsc=DEFAULT_GSC,
            sprout_mechanism=DEFAULT_SPROUT_COND,
            options=options,
        )

        deme_tree = DemeTree(tree_config)
        # Initially root is not hibernating.
        self.assertFalse(deme_tree.root._hibernating)
        self.assertTrue("hibernation" in deme_tree.config.options)
        self.assertTrue(deme_tree.config.options["hibernation"])
        deme_tree.run()
        # Root should be hibernated at some point.
        self.assertTrue(deme_tree.root._hibernating)
