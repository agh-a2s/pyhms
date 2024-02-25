import unittest

import numpy as np
from leap_ec.individual import Individual
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep import create_real_vector
from leap_ec.representation import Representation
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.problem import EvalCountingProblem, PrecisionCutoffProblem
from pyhms.stop_conditions import (
    AllStopped,
    DontStop,
    FitnessEvalLimitReached,
    GlobalStopCondition,
    NoActiveNonrootDemes,
    RootStopped,
    SingularProblemEvalLimitReached,
    SingularProblemPrecisionReached,
    WeightingStrategy,
)
from pyhms.tree import DemeTree

from .config import DEFAULT_SPROUT_COND, SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN

POPULATION_SIZE = 50


class TestGlobalStopCondition(unittest.TestCase):
    def get_deme_tree(self, gsc: GlobalStopCondition, problem: FunctionProblem | None = SQUARE_PROBLEM) -> DemeTree:
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=problem,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=problem,
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
        problem = EvalCountingProblem(SQUARE_PROBLEM)
        deme_tree = self.get_deme_tree(gsc, problem)
        # Evaluate problem LIMIT - 1 times:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=LIMIT - problem.n_evaluations - 1, problem=problem)
        Individual.evaluate_population(population)
        self.assertFalse(gsc(deme_tree))
        # Evaluate problem 1 more time:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=1, problem=problem)
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

    def test_fitness_eval_limit_reached(self):
        LIMIT = 100
        gsc = FitnessEvalLimitReached(LIMIT, WeightingStrategy.EQUAL)
        problem_level1 = EvalCountingProblem(SQUARE_PROBLEM)
        problem_level2 = EvalCountingProblem(SQUARE_PROBLEM)
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=problem_level1,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=problem_level2,
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
        # Evaluate problem (level 1) LIMIT - 1 times:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(
            pop_size=LIMIT - problem_level1.n_evaluations - 1, problem=problem_level1
        )
        Individual.evaluate_population(population)
        self.assertFalse(gsc(deme_tree))
        # Evaluate problem (level 2) 1 more time:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=1, problem=problem_level2)
        Individual.evaluate_population(population)
        self.assertTrue(gsc(deme_tree))

    def test_fitness_eval_limit_reached_root_only_strategy(self):
        LIMIT = 100
        gsc = FitnessEvalLimitReached(LIMIT, WeightingStrategy.ROOT)
        problem_level1 = EvalCountingProblem(SQUARE_PROBLEM)
        problem_level2 = EvalCountingProblem(SQUARE_PROBLEM)
        levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=problem_level1,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=problem_level2,
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
        # Evaluate problem (level 1) LIMIT - 1 times:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(
            pop_size=LIMIT - problem_level1.n_evaluations - 1, problem=problem_level1
        )
        Individual.evaluate_population(population)
        self.assertFalse(gsc(deme_tree))
        # Evaluate problem (level 2) 1 more time:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=1, problem=problem_level2)
        Individual.evaluate_population(population)
        self.assertFalse(gsc(deme_tree))
        # Evaluate problem (level 1) 1 more time:
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_PROBLEM_DOMAIN))
        population = representation.create_population(pop_size=1, problem=problem_level1)
        Individual.evaluate_population(population)
        self.assertTrue(gsc(deme_tree))
