import unittest

from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions.usc import dont_stop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM


class TestHibernation(unittest.TestCase):
    def test_deme_tree(self):
        options = {"hibernation": True}
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=[(-20, 20), (-20, 20)],
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=[(-20, 20), (-20, 20)],
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
        # Initially root is not hibernating.
        self.assertFalse(deme_tree.root._hibernating)
        self.assertTrue("hibernation" in deme_tree.config.options)
        self.assertTrue(deme_tree.config.options["hibernation"])
        deme_tree.run()
        # Root should be hibernated at some point.
        self.assertTrue(deme_tree.root._hibernating)