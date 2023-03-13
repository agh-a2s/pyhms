import numpy as np
from scipy.stats import truncnorm
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem

class NormalSampling(Sampling):
    def __init__(self, center: np.array, std_dev: float):
        super().__init__()
        self._center = center
        self._std_dev = std_dev

    def _do(self, problem: Problem, n_samples, **kwargs):
        distribution = truncnorm((problem.xl - self._center) / self._std_dev, (problem.xu - self._center) / self._std_dev, loc=self._center, scale=self._std_dev)
        return distribution.rvs([n_samples, problem.n_var])


def inject_population():
    pass