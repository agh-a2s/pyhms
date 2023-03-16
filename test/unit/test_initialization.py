import unittest

import numpy as np

from pyhms.core.problem import EvalCountingProblem
from pyhms.operators.initializers import NormalSampling

class UnitTestInitialization(unittest.TestCase):

    @staticmethod
    def getTestProblem():
        fit_fun = lambda _: 1
        problem = EvalCountingProblem(fit_fun, 1, -10, 10)
        return problem

    def test_if_truncated_normal_sampling_is_in_bounds(self):
        sampling = NormalSampling(center = np.array([-9]), std_dev = 2.5)
        problem = self.getTestProblem()
        population = sampling.do(problem, 10)
        out_of_bounds = False

        for ind in population:
            for dim in ind.get('X'):
                if dim < problem.xl or dim > problem.xu:
                    out_of_bounds = True
                    break

        self.assertFalse(out_of_bounds, "Normal sampling goes out of bounds")