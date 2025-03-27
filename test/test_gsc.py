import unittest

import numpy as np
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.core.individual import Individual
from pyhms.core.problem import PrecisionCutoffProblem, Problem
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.initializers import sample_uniform
from pyhms.stop_conditions import (
    AllStopped,
    DontStop,
    GlobalStopCondition,
    NoActiveNonrootDemes,
    RootStopped,
    SingularProblemEvalLimitReached,
    SingularProblemPrecisionReached,
)
from pyhms.tree import DemeTree

from .config import DEFAULT_SPROUT_COND, SQUARE_BOUNDS, SQUARE_PROBLEM

POPULATION_SIZE = 50


class TestGlobalStopCondition(unittest.TestCase):
    def get_deme_tree(self, gsc: GlobalStopCondition, problem: Problem | None = SQUARE_PROBLEM) -> DemeTree:
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=problem,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=problem,
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

    def test_root_stopped(self):
        gsc = RootStopped()
        deme_tree = self.get_deme_tree(gsc)
        self.assertFalse(gsc(deme_tree))
        deme_tree.root._active = False
        self.assertTrue(gsc(deme_tree))

    def test_all_stopped(self):
        gsc = AllStopped()
        deme_tree = self.get_deme_tree(gsc)
        self.assertFalse(gsc(deme_tree))
        initial_number_of_active_demes = len(deme_tree.active_demes)
        deme_tree.run_metaepoch()
        deme_tree.run_sprout()
        self.assertGreater(len(deme_tree.active_demes), initial_number_of_active_demes)
        for _, deme in deme_tree.active_demes:
            deme._active = False
        self.assertTrue(gsc(deme_tree))

    def test_no_active_nonroot_demes(self):
        gsc = NoActiveNonrootDemes(n_metaepochs=2)
        deme_tree = self.get_deme_tree(gsc)
        self.assertFalse(gsc(deme_tree))
        # Run first metaepoch:
        deme_tree.run_metaepoch()
        deme_tree.run_sprout()
        deme_tree.metaepoch_count += 1
        # Set all demes to inactive:
        for _, deme in deme_tree.active_demes:
            deme._active = False
        self.assertFalse(gsc(deme_tree))
        # Run second metaepoch:
        deme_tree.run_metaepoch()
        deme_tree.run_sprout()
        deme_tree.metaepoch_count += 1
        self.assertFalse(gsc(deme_tree))
        # Run thirs metaepoch:
        deme_tree.run_metaepoch()
        deme_tree.run_sprout()
        deme_tree.metaepoch_count += 1
        self.assertTrue(gsc(deme_tree))

    def test_singular_problem_eval_limit_reached(self):
        LIMIT = 100
        gsc = SingularProblemEvalLimitReached(LIMIT)
        deme_tree = self.get_deme_tree(gsc, SQUARE_PROBLEM)
        # Evaluate problem LIMIT - 1 times:
        population = Individual.create_population(
            pop_size=LIMIT - deme_tree.root._problem.n_evaluations - 1,
            problem=deme_tree.root._problem,
            initialize=sample_uniform(bounds=SQUARE_BOUNDS),
        )
        Individual.evaluate_population(population)
        self.assertFalse(gsc(deme_tree))
        # Evaluate problem 1 more time:
        population = Individual.create_population(
            pop_size=1,
            problem=deme_tree.root._problem,
            initialize=sample_uniform(bounds=SQUARE_BOUNDS),
        )
        Individual.evaluate_population(population)
        self.assertTrue(gsc(deme_tree))

    def test_singular_problem_precision_reached(self):
        problem = PrecisionCutoffProblem(SQUARE_PROBLEM, global_optima=0.0, precision=1e-15)
        gsc = SingularProblemPrecisionReached(problem)
        deme_tree = self.get_deme_tree(gsc, problem)
        self.assertFalse(gsc(deme_tree))
        individual = Individual(genome=np.array([0, 0]), problem=problem)
        individual.evaluate()
        self.assertTrue(gsc(deme_tree))
