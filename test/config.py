import os

import numpy as np
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.core.problem import FunctionProblem
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import get_NBC_sprout, get_simple_sprout
from pyhms.stop_conditions import DontStop, MetaepochLimit

DEFAULT_CENTERS = np.array([[-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]])


def square(x: np.ndarray) -> float:
    return sum(x**2)


class FunnelProblem:
    def __init__(self, centers: np.ndarray | None = DEFAULT_CENTERS):
        self.centers = centers

    def __call__(self, x: np.ndarray) -> float:
        return np.min([np.sum((x - center) ** 2) for center in self.centers])


four_funnels = FunnelProblem()


SQUARE_BOUNDS = np.array([(-20, 20), (-20, 20)])
FOUR_FUNNEL_BOUNDS = np.array([(-10.0, 10.0)] * 2)

SQUARE_PROBLEM = FunctionProblem(square, maximize=False, bounds=SQUARE_BOUNDS)
FOUR_FUNNELS_PROBLEM = FunctionProblem(four_funnels, maximize=False, bounds=FOUR_FUNNEL_BOUNDS)

NEGATIVE_SQUARE_PROBLEM = FunctionProblem(
    fitness_function=lambda x: -1 * sum(x**2), maximize=True, bounds=SQUARE_BOUNDS
)
NEGATIVE_FOUR_FUNNELS_PROBLEM = FunctionProblem(
    fitness_function=lambda x: -1 * four_funnels(x), maximize=True, bounds=FOUR_FUNNEL_BOUNDS
)

DEFAULT_GSC = MetaepochLimit(limit=10)

DEFAULT_LSC = DontStop()

LEVEL_LIMIT = 4

DEFAULT_SPROUT_COND = get_simple_sprout(1.0, level_limit=LEVEL_LIMIT)
DEFAULT_NBC_SPROUT_COND = get_NBC_sprout(level_limit=LEVEL_LIMIT)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def get_default_tree_config() -> TreeConfig:
    levels = [
        EALevelConfig(
            ea_class=SEA,
            generations=2,
            problem=SQUARE_PROBLEM,
            pop_size=20,
            mutation_std=1.0,
            lsc=DEFAULT_LSC,
        ),
        CMALevelConfig(
            generations=4,
            problem=SQUARE_PROBLEM,
            sigma0=2.5,
            lsc=DEFAULT_LSC,
        ),
    ]
    return TreeConfig(levels, DEFAULT_GSC, DEFAULT_SPROUT_COND, options={})
