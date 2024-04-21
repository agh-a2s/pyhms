from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from leap_ec.problem import FunctionProblem

from .config import DEFAULT_OPTIONS, BaseLevelConfig, CMALevelConfig, EALevelConfig, Options, TreeConfig
from .demes.single_pop_eas.sea import SEA
from .logging_ import LoggingLevel, parse_log_level
from .problem import EvalCutoffProblem
from .sprout import get_NBC_sprout
from .stop_conditions import (
    DontStop,
    GlobalStopCondition,
    MetaepochLimit,
    SingularProblemEvalLimitReached,
    UniversalStopCondition,
)
from .tree import DemeTree

DEFAULT_MAX_FUN = 10000


def hms(
    level_config: List[BaseLevelConfig],
    gsc: GlobalStopCondition | UniversalStopCondition,
    sprout_cond,
    options: Options = DEFAULT_OPTIONS,
):
    config = TreeConfig(level_config, gsc, sprout_cond, options=options)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return hms_tree


@dataclass
class OptimizeResult:
    x: np.ndarray
    nfev: int
    fun: float
    nit: int


def minimize(
    fun: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    maxfun: int | None = None,
    maxiter: int | None = None,
    seed: int | None = None,
    log_level: str | LoggingLevel | None = LoggingLevel.WARNING,
) -> OptimizeResult:
    # If neither maxfun nor maxiter are specified, default to maxfun=10000.
    if maxfun is None and maxiter is None:
        maxfun = DEFAULT_MAX_FUN
    function_problem = FunctionProblem(fun, maximize=False)
    if maxfun is not None:
        function_problem = EvalCutoffProblem(function_problem, eval_cutoff=maxfun)
    gsc: GlobalStopCondition | UniversalStopCondition = (
        SingularProblemEvalLimitReached(maxfun) if maxfun is not None else MetaepochLimit(maxiter)
    )
    level_config = [
        EALevelConfig(
            ea_class=SEA,
            generations=1,
            problem=function_problem,
            bounds=bounds,
            pop_size=35,
            mutation_std=1.0,
            lsc=DontStop(),
        ),
        CMALevelConfig(
            generations=20,
            problem=function_problem,
            bounds=bounds,
            sigma0=1.0,
            lsc=MetaepochLimit(20),
        ),
    ]
    sprout_condition = get_NBC_sprout(level_limit=4)
    options: Options = {"random_seed": seed, "log_level": parse_log_level(log_level)}
    config = TreeConfig(level_config, gsc, sprout_condition, options=options)
    hms_tree = DemeTree(config)
    hms_tree.run()
    return OptimizeResult(
        x=hms_tree.best_individual.genome,
        nfev=hms_tree.n_evaluations,
        fun=hms_tree.best_individual.fitness,
        nit=hms_tree.metaepoch_count,
    )
