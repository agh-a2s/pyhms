import unittest

import numpy as np
from leap_ec.individual import Individual
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep import create_real_vector
from leap_ec.representation import Representation
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.problem import EvalCountingProblem, PrecisionCutoffProblem
from pyhms.stop_conditions.usc import (
    UniversalStopCondition,
    MetaepochLimit,
    DontRun,
    DontStop,
)
from pyhms.stop_conditions.usc import DontStop
from pyhms.tree import DemeTree

from .config import DEFAULT_SPROUT_COND, SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN

POPULATION_SIZE = 50


class TestUniversalStopCondition(unittest.TestCase):
    def get_deme_tree(self, gsc: UniversalStopCondition) -> DemeTree:
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
