import os

import numpy as np
from leap_ec.problem import FunctionProblem
from pyhms.sprout import get_NBC_sprout, get_simple_sprout
from pyhms.stop_conditions import MetaepochLimit

SQUARE_PROBLEM = FunctionProblem(lambda x: sum(x**2), maximize=False)
SQUARE_BOUNDS = np.array([(-20, 20), (-20, 20)])

NEGATIVE_SQUARE_PROBLEM = FunctionProblem(lambda x: -1 * sum(x**2), maximize=True)

SQUARE_PROBLEM_DOMAIN = np.array([(-20, 20), (-20, 20)])

DEFAULT_GSC = MetaepochLimit(limit=10)

LEVEL_LIMIT = 4

DEFAULT_SPROUT_COND = get_simple_sprout(1.0, level_limit=LEVEL_LIMIT)
DEFAULT_NBC_SPROUT_COND = get_NBC_sprout(level_limit=LEVEL_LIMIT)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
