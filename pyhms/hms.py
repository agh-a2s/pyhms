from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .config import DEFAULT_OPTIONS, BaseLevelConfig, CMALevelConfig, EALevelConfig, Options, TreeConfig
from .core.problem import EvalCutoffProblem, FunctionProblem
from .demes.single_pop_eas.sea import SEA
from .logging_ import LoggingLevel, parse_log_level
from .sprout import get_NBC_sprout
from .stop_conditions import (
    DontStop,
    FitnessSteadiness,
    GlobalStopCondition,
    MetaepochLimit,
    SingularProblemEvalLimitReached,
    UniversalStopCondition,
)
from .tree import DemeTree
from .utils.parameter_initializer import get_default_generations, get_default_mutation_std, get_default_population_size

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
    bounds: np.ndarray | list[tuple[float, float]],
    maxfun: int | None = None,
    maxiter: int | None = None,
    seed: int | None = None,
    log_level: str | LoggingLevel | None = LoggingLevel.WARNING,
) -> OptimizeResult:
    if isinstance(bounds, list):
        bounds = np.array(bounds)
    # If neither maxfun nor maxiter are specified, default to maxfun=DEFAULT_MAX_FUN.
    if maxfun is None and maxiter is None:
        maxfun = DEFAULT_MAX_FUN
    function_problem = FunctionProblem(fun, maximize=False, bounds=bounds)
    wrapped_function_problem = EvalCutoffProblem(function_problem, eval_cutoff=maxfun) if maxfun else function_problem
    gsc: GlobalStopCondition | UniversalStopCondition = (
        SingularProblemEvalLimitReached(maxfun) if maxfun is not None else MetaepochLimit(maxiter)
    )
    level_config = [
        EALevelConfig(
            ea_class=SEA,
            generations=get_default_generations(bounds, tree_level=0),
            problem=wrapped_function_problem,
            pop_size=get_default_population_size(bounds, tree_level=0),
            mutation_std=get_default_mutation_std(bounds, tree_level=0),
            lsc=DontStop(),
        ),
        CMALevelConfig(
            generations=get_default_generations(bounds, tree_level=1),
            problem=wrapped_function_problem,
            sigma0=None,
            lsc=FitnessSteadiness(),
        ),
    ]
    sprout_condition = get_NBC_sprout()
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
