import unittest

import numpy as np
from pyhms.core.individual import Individual
from pyhms.utils.r5s import R5SSelection
from .config import SQUARE_PROBLEM


class TestR5S(unittest.TestCase):
    def test_r5s(self):
        genomes = np.array(
            [
                [0.1, 0.2],
                [0.4, 0.5],
                [0.7, 0.8],
                [0.2, 0.3],
                [0.6, 0.1],
                [0.8, 0.5],
                [0.3, 0.9],
                [0.9, 0.3],
                [0.05, 0.9],
                [0.5, 0.2],
                [0.6, 0.6],
                [-0.15, 0.85],
            ]
        )
        fitness_values = np.array(
            [0.1, 0.5, 0.9, 0.3, 0.6, 0.8, 0.4, 0.95, 0.25, 0.7, 0.55, 0.45]
        )
        individuals = [
            # The fitness values are not real, they are just for testing purposes.
            # R5SSelection doesn't use the problem to evaluate the individuals.
            Individual(genome=genome, problem=SQUARE_PROBLEM, fitness=fitness)
            for genome, fitness in zip(genomes, fitness_values)
        ]
        selection = R5SSelection()
        selected_individuals = selection(individuals)
        true_fitness_values = [0.1, 0.25, 0.5, 0.6]
        self.assertEqual(
            set([individual.fitness for individual in selected_individuals]),
            set(true_fitness_values),
        )
