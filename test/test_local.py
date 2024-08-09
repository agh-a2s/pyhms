import unittest

from pyhms.config import EALevelConfig, LocalOptimizationConfig, TreeConfig
from pyhms.core.initializers import InjectionInitializer, UniformGlobalInitializer
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions import DontStop
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM


class TestLocalOptimization(unittest.TestCase):
    def test_deme_tree(self):
        options = {"hibernation": False, "log_level": "debug", "random_seed": 1}
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=SQUARE_PROBLEM,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
                pop_initializer=UniformGlobalInitializer,
            ),
            LocalOptimizationConfig(
                problem=SQUARE_PROBLEM,
                lsc=DontStop(),
                pop_initializer=InjectionInitializer,
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
        self.assertTrue(deme_tree.root.best_current_individual.fitness > deme_tree.best_leaf_individual.fitness)
