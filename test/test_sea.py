import unittest

import numpy as np
from pyhms.core.individual import Individual
from pyhms.core.population import Population
from pyhms.core.problem import EvalCountingProblem
from pyhms.demes.single_pop_eas.sea import GaussianMutation
from .config import SQUARE_PROBLEM, SQUARE_BOUNDS


class TestVariationalOperators(unittest.TestCase):
    def test_gaussian_mutation(self):
        np.random.seed(0)
        population_size = 10000
        probability = 0.1
        mutation_std = 1.0
        genomes = np.random.uniform(
            SQUARE_BOUNDS[:, 0], SQUARE_BOUNDS[:, 1], (population_size, 2)
        )
        individuals = [
            Individual(
                genome=genome,
                problem=SQUARE_PROBLEM,
            )
            for genome in genomes
        ]
        Individual.evaluate_population(individuals)
        population = Population.from_individuals(individuals)
        mutation = GaussianMutation(mutation_std, SQUARE_BOUNDS, probability)
        mutated_population = mutation(population)
        self.assertEqual(population.size, mutated_population.size)
        changed_genomes_frequency = np.mean(
            mutated_population.genomes != population.genomes
        )
        self.assertTrue(np.abs(changed_genomes_frequency - probability) < 1e-3)
        self.assertTrue(np.all(SQUARE_BOUNDS[:, 0] <= mutated_population.genomes))
        self.assertTrue(np.all(mutated_population.genomes <= SQUARE_BOUNDS[:, 1]))
        self.assertFalse(np.array_equal(population.genomes, mutated_population.genomes))
        self.assertFalse(np.isnan(mutated_population.fitnesses).any())
        average_gene_change = np.mean(mutated_population.genomes - population.genomes)
        self.assertTrue(np.abs(average_gene_change) < 1e-2)
