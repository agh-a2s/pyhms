import unittest

import numpy as np
from pyhms.core.population import Population
from pyhms.core.individual import Individual
from .config import SQUARE_PROBLEM
from pyhms.core.problem import EvalCountingProblem


class TestPopulation(unittest.TestCase):
    def test_population_methods(self):
        genomes = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
        problem = EvalCountingProblem(SQUARE_PROBLEM)
        individuals = [
            Individual(
                genome=genome,
                problem=problem,
            )
            for genome in genomes
        ]
        Individual.evaluate_population(individuals)
        population = Population.from_individuals(individuals)
        self.assertTrue(np.array_equal(population.genomes, genomes))
        self.assertTrue(np.array_equal(population.fitnesses, np.array([0.0, 2.0])))
        self.assertEqual(population.size, len(genomes))
        self.assertListEqual(population.to_individuals(), individuals)
        self.assertTrue(np.array_equal(population.topk(1).genomes[0], genomes[0]))
        new_genomes = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        population_copy = population.copy()
        population.update_genome(new_genomes)
        self.assertFalse(np.array_equal(population.genomes, population_copy.genomes))
        self.assertTrue(np.array_equal(population.genomes, new_genomes))
        self.assertTrue(
            np.array_equal(
                population.fitnesses, np.array([np.nan, 2.0]), equal_nan=True
            )
        )
        population.evaluate()
        self.assertEqual(problem.n_evaluations, len(genomes) + 1)
