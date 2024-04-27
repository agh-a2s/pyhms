import unittest

import numpy as np
from pyhms.core.individual import Individual
from pyhms.core.population import Population
from pyhms.demes.single_pop_eas.sea import ArithmeticCrossover, GaussianMutation, UniformMutation

from .config import SQUARE_BOUNDS, SQUARE_PROBLEM


class TestVariationalOperators(unittest.TestCase):
    ABS_TOL = 1e-2

    @classmethod
    def get_initial_population(cls) -> Population:
        population_size = 10000
        genomes = np.random.uniform(SQUARE_BOUNDS[:, 0], SQUARE_BOUNDS[:, 1], (population_size, 2))
        individuals = [
            Individual(
                genome=genome,
                problem=SQUARE_PROBLEM,
            )
            for genome in genomes
        ]
        Individual.evaluate_population(individuals)
        return Population.from_individuals(individuals)

    def test_gaussian_mutation(self):
        np.random.seed(0)
        probability = 0.1
        mutation_std = 1.0
        population = self.get_initial_population()
        mutation = GaussianMutation(mutation_std, SQUARE_BOUNDS, probability)
        mutated_population = mutation(population)
        self.assertEqual(population.size, mutated_population.size)
        changed_genomes_frequency = np.mean(mutated_population.genomes != population.genomes)
        self.assertTrue(np.abs(changed_genomes_frequency - probability) < self.ABS_TOL)
        self.assertTrue(np.all(SQUARE_BOUNDS[:, 0] <= mutated_population.genomes))
        self.assertTrue(np.all(mutated_population.genomes <= SQUARE_BOUNDS[:, 1]))
        self.assertFalse(np.array_equal(population.genomes, mutated_population.genomes))
        self.assertFalse(np.isnan(mutated_population.fitnesses).any())
        average_gene_change = np.mean(mutated_population.genomes - population.genomes)
        self.assertTrue(np.abs(average_gene_change) < self.ABS_TOL)

    def test_uniform_mutation(self):
        np.random.seed(0)
        probability = 0.1
        population = self.get_initial_population()
        mutation = UniformMutation(SQUARE_BOUNDS, probability)
        mutated_population = mutation(population)
        self.assertEqual(population.size, mutated_population.size)
        changed_genomes_frequency = np.mean(mutated_population.genomes != population.genomes)
        self.assertTrue(np.abs(changed_genomes_frequency - probability) < self.ABS_TOL)
        self.assertTrue(np.all(SQUARE_BOUNDS[:, 0] <= mutated_population.genomes))
        self.assertTrue(np.all(mutated_population.genomes <= SQUARE_BOUNDS[:, 1]))
        self.assertFalse(np.array_equal(population.genomes, mutated_population.genomes))
        self.assertFalse(np.isnan(mutated_population.fitnesses).any())

    def test_arithmetic_crossover(self):
        np.random.seed(0)
        genomes = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        population = Population(
            genomes=genomes,
            fitnesses=np.array([5.0, 25.0]),
            problem=SQUARE_PROBLEM,
        )
        crossover = ArithmeticCrossover(probability=1.0, evaluate_fitness=True)
        offspring_population = crossover(population)
        self.assertEqual(population.size, offspring_population.size)
        # We calculate alpha from the offspring population and the original genomes:
        alpha = (offspring_population.genomes[0][0] - genomes[1][0]) / (genomes[0][0] - genomes[1][0])
        self.assertTrue(
            np.array_equal(
                offspring_population.genomes[1],
                genomes[1] * alpha + genomes[0] * (1 - alpha),
            )
        )
        self.assertFalse(np.isnan(offspring_population.fitnesses).any())

    def test_arithmetic_crossover_odd_number_of_individuals(self):
        np.random.seed(0)
        genomes = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        population = Population(
            genomes=genomes,
            fitnesses=np.array([5.0, 25.0, 81.0]),
            problem=SQUARE_PROBLEM,
        )
        crossover = ArithmeticCrossover(probability=1.0, evaluate_fitness=False)
        offspring_population = crossover(population)
        self.assertEqual(population.size, offspring_population.size)
        # We calculate alpha from the offspring population and the original genomes:
        alpha = (offspring_population.genomes[0][0] - genomes[1][0]) / (genomes[0][0] - genomes[1][0])
        self.assertTrue(
            np.array_equal(
                offspring_population.genomes[1],
                genomes[1] * alpha + genomes[0] * (1 - alpha),
            )
        )
        self.assertTrue(np.array_equal(offspring_population.genomes[2], genomes[2]))
