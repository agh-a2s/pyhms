import unittest
from copy import deepcopy

import numpy as np
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.core.individual import Individual
from pyhms.core.initializers import InjectionInitializer, UniformGlobalInitializer
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout.sprout_filters import DemeLimit, FarEnough, LevelLimit, NBC_FarEnough
from pyhms.sprout.sprout_generators import BestPerDeme, NBC_Generator
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.stop_conditions import DontStop
from pyhms.tree import DemeTree

from .config import FOUR_FUNNELS_PROBLEM, NEGATIVE_FOUR_FUNNELS_PROBLEM


class TestSprout(unittest.TestCase):
    def setUp(self):
        genome = np.array(
            [
                [5.0, 5.0],
                [5.0, 5.25],
                [-5.0, -5.5],
                [-7.438652231484955, 1.6694449410425545],
                [8.868488721370134, -3.8092004083209963],
                [-2.3623991956204105, 3.540653675138808],
                [4.46, -4.49],
                [-2.5619700848072258, 2.240573438147379],
                [2.4633250549304986, 5.347669720933412],
                [-7.703343438597178, -4.0488667762712645],
                [0.9313567871638302, 6.166115779099393],
                [2.415406130753734, -6.885780821947341],
                [5.749793158123332, 6.091968174818984],
                [9.868256923059192, -0.856544263545203],
                [8.461437828094471, -7.69756109622826],
                [0.4762125269013602, -2.5571632669301643],
                [2.891842394903989, -5.41814403371089],
                [2.285558012997315, 3.403792214048396],
                [-6.39, 2.88],
                [-9.177658322165303, -0.3086626198612663],
            ]
        )
        self.best_gene_indices = [0, 2, 6, 18]
        self.best_gene = genome[self.best_gene_indices[0]]
        self.second_best_gene = genome[self.best_gene_indices[1]]
        self.third_best_gene = genome[self.best_gene_indices[2]]

        self.minimize_problem = FOUR_FUNNELS_PROBLEM
        self.maximize_problem = NEGATIVE_FOUR_FUNNELS_PROBLEM
        self.initial_population_min = [
            Individual(gene, self.minimize_problem, self.minimize_problem.evaluate(gene)) for gene in genome
        ]
        self.initial_population_max = [
            Individual(gene, self.maximize_problem, self.maximize_problem.evaluate(gene)) for gene in genome
        ]

        minimize_levels = [
            EALevelConfig(
                ea_class=SEA,
                generations=1,
                problem=self.minimize_problem,
                pop_size=20,
                mutation_std=1.0,
                lsc=DontStop(),
                pop_initializer_type=UniformGlobalInitializer,
            ),
            CMALevelConfig(
                generations=4,
                problem=self.minimize_problem,
                sigma0=0.01,
                lsc=DontStop(),
                pop_initializer_type=InjectionInitializer,
            ),
        ]
        maximize_levels = deepcopy(minimize_levels)
        maximize_levels[0].problem = self.maximize_problem
        maximize_levels[1].problem = self.maximize_problem
        self.minimize_config = TreeConfig(minimize_levels, DontStop(), SproutMechanism([], [], []), options={})
        self.maximize_config = TreeConfig(maximize_levels, DontStop(), SproutMechanism([], [], []), options={})

    def test_simple_sprout_candidates(self):
        tree = DemeTree(self.minimize_config)
        sprout = SproutMechanism(BestPerDeme(), [FarEnough(1.0, 2)], [LevelLimit(2)])

        tree._sprout_mechanism = sprout
        tree.root._history = [[self.initial_population_min]]
        first_generated = sprout.candidates_generator(tree)
        self.assertTrue(np.all(first_generated[tree.root].individuals[0].genome == self.best_gene))
        first_candidates = sprout.apply_deme_filters(first_generated, tree)
        first_candidates = sprout.apply_tree_filters(first_candidates, tree)
        self.assertTrue(np.all(first_candidates[tree.root].individuals[0].genome == self.best_gene))

        tree._do_sprout(first_candidates)
        second_generated = sprout.candidates_generator(tree)
        second_candidates = sprout.apply_deme_filters(second_generated, tree)
        self.assertEqual(len(second_candidates[tree.root].individuals), 0)

    def test_nbc_sprout_candidates(self):
        tree = DemeTree(self.minimize_config)
        sprout = SproutMechanism(
            NBC_Generator(2.0, 0.8),
            [NBC_FarEnough(2.0, 2), DemeLimit(1)],
            [LevelLimit(2)],
        )

        tree._sprout_mechanism = sprout
        tree.root._history = [[self.initial_population_min]]
        first_generated = sprout.candidates_generator(tree)
        self.assertTrue(
            np.all(
                np.array([ind.genome for ind in first_generated[tree.root].individuals])
                == np.array([self.initial_population_min[i].genome for i in self.best_gene_indices])
            )
        )
        first_candidates = sprout.apply_deme_filters(first_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in first_candidates[tree.root].individuals]) == self.best_gene)
        )

        tree._do_sprout(first_candidates)
        second_generated = sprout.candidates_generator(tree)
        second_candidates = sprout.apply_deme_filters(second_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in second_generated[tree.root].individuals]) == self.second_best_gene)
        )

        tree._do_sprout(second_candidates)
        third_generated = sprout.candidates_generator(tree)
        third_candidates = sprout.apply_deme_filters(third_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in third_candidates[tree.root].individuals]) == self.third_best_gene)
        )
        third_candidates = sprout.apply_tree_filters(third_generated, tree)
        self.assertEqual(len(third_candidates[tree.root].individuals), 0)

    def test_simple_sprout_candidates_maximization(self):
        tree = DemeTree(self.maximize_config)
        sprout = SproutMechanism(BestPerDeme(), [FarEnough(1.0, 2)], [LevelLimit(2)])

        tree._sprout_mechanism = sprout
        tree.root._history = [[self.initial_population_max]]
        first_generated = sprout.candidates_generator(tree)
        self.assertTrue(np.all(first_generated[tree.root].individuals[0].genome == self.best_gene))
        first_candidates = sprout.apply_deme_filters(first_generated, tree)
        first_candidates = sprout.apply_tree_filters(first_candidates, tree)
        self.assertTrue(np.all(first_candidates[tree.root].individuals[0].genome == self.best_gene))

        tree._do_sprout(first_candidates)
        second_generated = sprout.candidates_generator(tree)
        second_candidates = sprout.apply_deme_filters(second_generated, tree)
        self.assertEqual(len(second_candidates[tree.root].individuals), 0)

    def test_nbc_sprout_candidates_maximization(self):
        tree = DemeTree(self.maximize_config)
        sprout = SproutMechanism(
            NBC_Generator(2.0, 0.8),
            [NBC_FarEnough(2.0, 2), DemeLimit(1)],
            [LevelLimit(2)],
        )

        tree._sprout_mechanism = sprout
        tree.root._history = [[self.initial_population_max]]
        first_generated = sprout.candidates_generator(tree)
        self.assertTrue(
            np.all(
                np.array([ind.genome for ind in first_generated[tree.root].individuals])
                == np.array([self.initial_population_min[i].genome for i in self.best_gene_indices])
            )
        )
        first_candidates = sprout.apply_deme_filters(first_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in first_candidates[tree.root].individuals]) == self.best_gene)
        )

        tree._do_sprout(first_candidates)
        second_generated = sprout.candidates_generator(tree)
        second_candidates = sprout.apply_deme_filters(second_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in second_generated[tree.root].individuals]) == self.second_best_gene)
        )

        tree._do_sprout(second_candidates)
        third_generated = sprout.candidates_generator(tree)
        third_candidates = sprout.apply_deme_filters(third_generated, tree)
        self.assertTrue(
            np.all(np.array([ind.genome for ind in third_candidates[tree.root].individuals]) == self.third_best_gene)
        )
        third_candidates = sprout.apply_tree_filters(third_generated, tree)
        self.assertEqual(len(third_candidates[tree.root].individuals), 0)
