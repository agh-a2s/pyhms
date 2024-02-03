from pyhms.initializers import sample_normal
import unittest
import numpy as np


class TestInitializers(unittest.TestCase):
    def test_sample_normal(self):
        np.random.seed(0)
        N = 10
        x = np.zeros(N)
        std_dev = 1.0
        bounds = [(-5, 5)] * N
        create = sample_normal(x, std_dev, bounds)
        population = np.array([create() for _ in range(10000)])
        population_mean = np.mean(population, axis=0)
        population_std = np.std(population, axis=0)
        np.testing.assert_allclose(population_mean, x, rtol=1e-2, atol=1e-1)
        np.testing.assert_allclose(population_std, std_dev, rtol=1e-2, atol=1e-1)
