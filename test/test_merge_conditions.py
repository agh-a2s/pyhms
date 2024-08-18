import numpy as np
import unittest
from scipy.optimize import Bounds
from pyhms.cluster.merge_conditions import (
    Cluster,
    LocalOptimizationMergeCondition,
    Problem,
)


class TestLocalOptimizationMergeCondition(unittest.TestCase):
    def test_can_merge(self):
        problem = Problem()  # Replace with your actual problem instance
        merge_condition = LocalOptimizationMergeCondition(problem)

        cluster1 = Cluster(
            population=None,  # Replace with your actual population instance
            mean=np.array([1, 2]),  # Replace with your actual mean array
            covariance_matrix=np.array(
                [[1, 0], [0, 1]]
            ),  # Replace with your actual covariance matrix
        )
        cluster2 = Cluster(
            population=None,  # Replace with your actual population instance
            mean=np.array([3, 4]),  # Replace with your actual mean array
            covariance_matrix=np.array(
                [[1, 0], [0, 1]]
            ),  # Replace with your actual covariance matrix
        )

        can_merge = merge_condition.can_merge(cluster1, cluster2)

        self.assertTrue(can_merge)
