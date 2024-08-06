import unittest
from pyhms.merge_conditions import MergeCondition
import numpy as np


class TestMergeCondition(unittest.TestCase):

    def test_find_closest_points(self):
        cluster1 = np.array([[-1, -2], [0, 0], [-5, -6]])
        cluster2 = np.array([[7, 8], [9, 10], [2, 1]])

        closest_point1, closest_point2 = MergeCondition.find_closest_points(
            cluster1, cluster2
        )

        np.testing.assert_array_equal(closest_point1, [0, 0])
        np.testing.assert_array_equal(closest_point2, [2, 1])
