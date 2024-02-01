import numpy as np
from leap_ec.problem import FunctionProblem
from pyhms.sprout import get_simple_sprout
from pyhms.stop_conditions.usc import metaepoch_limit

SQUARE_PROBLEM = FunctionProblem(lambda x: sum(x**2), maximize=False)

NEGATIVE_SQUARE_PROBLEM = FunctionProblem(lambda x: -1 * sum(x**2), maximize=True)

SQUARE_PROBLEM_DOMAIN = np.array([(-20, 20), (-20, 20)])

DEFAULT_GSC = metaepoch_limit(limit=10)

DEFAULT_SPROUT_COND = get_simple_sprout(1.0)
