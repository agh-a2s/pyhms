import os

import numpy as np
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.core.problem import FunctionProblem
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import get_NBC_sprout, get_simple_sprout
from pyhms.stop_conditions import DontStop, MetaepochLimit


def square(x: np.ndarray) -> float:
    return sum(x**2)


SQUARE_PROBLEM = FunctionProblem(square, maximize=False)
SQUARE_BOUNDS = np.array([(-20, 20), (-20, 20)])

SQUARE_PROBLEM = FunctionProblem(fitness_function=lambda x: sum(x**2), maximize=False, bounds=SQUARE_BOUNDS)

NEGATIVE_SQUARE_PROBLEM = FunctionProblem(
    fitness_function=lambda x: -1 * sum(x**2), maximize=True, bounds=SQUARE_BOUNDS
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
            bounds=SQUARE_BOUNDS,
            pop_size=20,
            mutation_std=1.0,
            lsc=DEFAULT_LSC,
        ),
        CMALevelConfig(
            generations=4,
            problem=SQUARE_PROBLEM,
            bounds=SQUARE_BOUNDS,
            sigma0=2.5,
            lsc=DEFAULT_LSC,
        ),
    ]
    return TreeConfig(levels, DEFAULT_GSC, DEFAULT_SPROUT_COND, options={})
