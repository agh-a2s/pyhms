import unittest

import numpy as np
from leap_ec.real_rep import create_real_vector
from leap_ec.representation import Representation
from pyhms.core.individual import Individual
from pyhms.problem import EvalCountingProblem, EvalCutoffProblem, PrecisionCutoffProblem, StatsGatheringProblem

from .config import NEGATIVE_SQUARE_PROBLEM, SQUARE_BOUNDS, SQUARE_PROBLEM

POPULATION_SIZE = 50


class TestProblemDecorator(unittest.TestCase):
    def test_eval_counting_problem_for_one_individual(self):
        eval_counting_problem = EvalCountingProblem(SQUARE_PROBLEM)
        individual = Individual(
            genome=np.array([1.0, 1.0]),
            problem=eval_counting_problem,
        )
        self.assertIsNone(
            individual.fitness,
            "Fitness should be None before evaluation",
        )
        self.assertEqual(
            eval_counting_problem.n_evaluations,
            0,
        )
        individual.evaluate()
        self.assertEqual(
            individual.fitness,
            2.0,
            "Fitness should be set after evaluation",
        )
        self.assertEqual(
            eval_counting_problem.n_evaluations,
            1,
            "Problem should count evaluations",
        )

    def test_eval_counting_problem(self):
        eval_counting_problem = EvalCountingProblem(SQUARE_PROBLEM)
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_BOUNDS))
        population = representation.create_population(pop_size=POPULATION_SIZE, problem=eval_counting_problem)
        Individual.evaluate_population(population)
        self.assertEqual(
            eval_counting_problem.n_evaluations,
            POPULATION_SIZE,
            "Problem should count evaluations",
        )

    def test_eval_cutoff_problem_for_minimization(self):
        EVAL_CUTOFF = 10
        eval_cutoff_problem = EvalCutoffProblem(NEGATIVE_SQUARE_PROBLEM, EVAL_CUTOFF)
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_BOUNDS))
        population = representation.create_population(pop_size=EVAL_CUTOFF, problem=eval_cutoff_problem)
        Individual.evaluate_population(population)

        self.assertEqual(
            eval_cutoff_problem.n_evaluations,
            EVAL_CUTOFF,
            "Problem should count evaluations",
        )

        individual = Individual(
            genome=np.array([1.0, 1.0]),
            problem=eval_cutoff_problem,
        )
        individual.evaluate()
        self.assertEqual(
            individual.fitness,
            -np.inf,
            "Problem should stop evaluating after reaching the cutoff",
        )

    def test_eval_cutoff_problem_for_maximization(self):
        EVAL_CUTOFF = 10
        eval_cutoff_problem = EvalCutoffProblem(SQUARE_PROBLEM, EVAL_CUTOFF)
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_BOUNDS))
        population = representation.create_population(pop_size=EVAL_CUTOFF, problem=eval_cutoff_problem)
        Individual.evaluate_population(population)

        self.assertEqual(
            eval_cutoff_problem.n_evaluations,
            EVAL_CUTOFF,
            "Problem should count evaluations",
        )

        individual = Individual(
            genome=np.array([1.0, 1.0]),
            problem=eval_cutoff_problem,
        )
        individual.evaluate()
        self.assertEqual(
            individual.fitness,
            np.inf,
            "Problem should stop evaluating after reaching the cutoff",
        )

    def test_precision_cutoff_problem(self):
        PRECISION_CUTOFF = 1e-3
        SQUARE_GLOBAL_OPTIMA = 0.0
        precision_cutoff_problem = PrecisionCutoffProblem(SQUARE_PROBLEM, SQUARE_GLOBAL_OPTIMA, PRECISION_CUTOFF)
        individual = Individual(
            genome=np.array([1.0, 1.0]),
            problem=precision_cutoff_problem,
        )
        individual.evaluate()
        self.assertEqual(
            precision_cutoff_problem.ETA,
            np.inf,
        )
        self.assertEqual(
            precision_cutoff_problem.n_evaluations,
            1,
        )
        self.assertFalse(
            precision_cutoff_problem.hit_precision,
        )
        individual_close_to_global_optima = Individual(
            genome=np.array([1e-5, 1e-5]),
            problem=precision_cutoff_problem,
        )
        individual_close_to_global_optima.evaluate()
        self.assertEqual(
            precision_cutoff_problem.ETA,
            2,
            "Problem should save ETA after reaching the cutoff",
        )
        self.assertEqual(
            precision_cutoff_problem.n_evaluations,
            2,
            "Problem should count evaluations",
        )
        self.assertTrue(
            precision_cutoff_problem.hit_precision,
            "Problem should save that precision was hit",
        )

    def test_stats_gathering_problem(self):
        stats_gathering_problem = StatsGatheringProblem(SQUARE_PROBLEM)
        representation = Representation(initialize=create_real_vector(bounds=SQUARE_BOUNDS))
        population = representation.create_population(pop_size=POPULATION_SIZE, problem=stats_gathering_problem)
        Individual.evaluate_population(population)
        self.assertEqual(
            stats_gathering_problem.n_evaluations,
            POPULATION_SIZE,
            "Problem should count evaluations",
        )
        self.assertEqual(
            len(stats_gathering_problem.durations),
            POPULATION_SIZE,
            "Problem should measure durations",
        )
        mean, std = stats_gathering_problem.duration_stats
        self.assertGreater(
            mean,
            0,
            "Mean duration should be greater than 0",
        )
        self.assertGreater(
            std,
            0,
            "Standard deviation of durations should be greater than 0",
        )
