import unittest

import numpy as np
from pyhms.utils.covariance_estimate import estimate_sigma0, find_closest_rows, get_initial_sigma0_from_bounds


class TestCovarianceEstimateUtils(unittest.TestCase):
    def test_find_closest_rows(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2])
        closest_rows = find_closest_rows(X, y, 1)
        self.assertTrue(np.array_equal(closest_rows[0], y))

        closest_rows = find_closest_rows(X, y, 2)
        self.assertTrue(np.array_equal(closest_rows, np.array([[1, 2], [3, 4]])))

    def test_estimate_sigma0(self):
        X = np.array([[92, 80], [60, 30], [100, 70]])
        sigma0 = estimate_sigma0(X)
        self.assertAlmostEqual(sigma0, 23.9582971014)

    def test_sigma0_from_bounds(self):
        bounds = np.array([[0, 1], [0, 1]])
        sigma0 = get_initial_sigma0_from_bounds(bounds)
        self.assertAlmostEqual(sigma0, 1 / 6)
