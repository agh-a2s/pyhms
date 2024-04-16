import unittest

import numpy as np
from pyhms.utils.covariance_estimate import (
    estimate_covariance,
    estimate_sigma0,
    estimate_stds,
    find_closest_rows,
    get_initial_sigma0_from_bounds,
)


class TestCovarianceEstimateUtils(unittest.TestCase):
    def test_find_closest_rows(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2])
        closest_rows = find_closest_rows(X, y, 1)
        self.assertTrue(np.array_equal(closest_rows[0], y))

        closest_rows = find_closest_rows(X, y, 2)
        self.assertTrue(np.array_equal(closest_rows, np.array([[1, 2], [3, 4]])))

    def test_estimate_covariance_sigma0_and_stds(self):
        np.random.seed(0)
        mean_vector = np.array([0, 0, 0])
        cov_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
        X = np.random.multivariate_normal(mean_vector, cov_matrix, 10000)
        cov_estimate = estimate_covariance(X)
        self.assertTrue(np.all(np.abs(cov_estimate - cov_matrix) < 5e-2))
        sigma0 = estimate_sigma0(X)
        self.assertTrue(np.abs(sigma0 - 1) < 1e-2)
        stds = estimate_stds(X)
        self.assertTrue(np.all(np.abs(stds - 1) < 5e-2))

    def test_sigma0_from_bounds(self):
        bounds = np.array([[0, 1], [0, 1]])
        sigma0 = get_initial_sigma0_from_bounds(bounds)
        self.assertAlmostEqual(sigma0, 1 / 6)
