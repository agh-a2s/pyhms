from pyhms.config import CMALevelConfig, EALevelConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import get_simple_sprout
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from leap_ec.problem import FunctionProblem

SQUARE_PROBLEM = FunctionProblem(lambda x: sum(x**2), maximize=False)

DEFAULT_GSC = metaepoch_limit(limit=10)

DEFAULT_SPROUT_COND = get_simple_sprout(1.0)

DEFAULT_LEVELS_CONFIG = [
    EALevelConfig(
        ea_class=SEA,
        generations=2,
        problem=SQUARE_PROBLEM,
        bounds=[(-20, 20), (-20, 20)],
        pop_size=20,
        mutation_std=1.0,
        lsc=dont_stop(),
    ),
    CMALevelConfig(
        generations=4,
        problem=SQUARE_PROBLEM,
        bounds=[(-20, 20), (-20, 20)],
        sigma0=2.5,
        lsc=dont_stop(),
    ),
]
