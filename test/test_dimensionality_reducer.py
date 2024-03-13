import unittest

import numpy as np
from pyhms.utils.visualisation.dimensionality_reduction import NaiveDimensionalityReducer


class TestNaiveDimensionalityReducer(unittest.TestCase):
    def test_fit_transform(self):
        X = np.random.rand(1000, 4)
        reducer = NaiveDimensionalityReducer()
        X_reduced = reducer.fit_transform(X)
        self.assertEqual(X_reduced.shape, (1000, 2))
        self.assertTrue(np.array_equal(X_reduced, X[:, :2]))
