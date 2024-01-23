from leap_ec.problem import FunctionProblem
from pyhms.sprout import get_simple_sprout
from pyhms.stop_conditions.usc import metaepoch_limit

SQUARE_PROBLEM = FunctionProblem(lambda x: sum(x**2), maximize=False)

DEFAULT_GSC = metaepoch_limit(limit=10)

DEFAULT_SPROUT_COND = get_simple_sprout(1.0)
