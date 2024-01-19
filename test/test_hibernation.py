import unittest

from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.tree import DemeTree
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import get_simple_sprout
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from leap_ec.problem import FunctionProblem
from .config import DEFAULT_LEVELS_CONFIG, DEFAULT_GSC, DEFAULT_SPROUT_COND


class TestHibernation(unittest.TestCase):
    def test_deme_tree(self):
        options = {"hibernation": True}
        levels = DEFAULT_LEVELS_CONFIG
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
