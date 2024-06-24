import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import dill as pkl
import cma
import csv
from dataclasses import dataclass
from multiprocess import Pool
import numpy as np

from pyhms.config import CMALevelConfig, DELevelConfig, SHADELevelConfig, TreeConfig
from pyhms.core.problem import EvalCutoffProblem, FunctionProblem
from pyhms.logging_ import LoggingLevel
from pyhms.sprout import get_NBC_sprout
from pyhms.stop_conditions import (
    DontStop,
    FitnessSteadiness,
    SingularProblemEvalLimitReached,
    FitnessEvalLimitReached,
)
from pyhms.tree import DemeTree
from pyhms.logging_ import LoggingLevel

from experiment.F2_diff_solver import prepare_stopping_costfun

with open('data_file.pkl', 'rb') as f:
    f_data = pkl.load(f)
TS = {opening: f_data['T{key}'.format(key=opening)] for opening in range(10,101,10)}
YMS = {opening: f_data['Y{key}'.format(key=opening)] for opening in range(10,101,10)}

U_SIZE = 10
X1_SIZE = 11

SMALL_U_SIZE = 5
SMALL_X1_SIZE = 4

Td_tile = np.array([0.1, 0.1, 0.1, 0.34, 0.55, 0.54, 0.52, 0.53, 0.41, 0.41])
A1_grid = np.tile(np.array([151.57, 89.4135, 36.847, 17.458, 10.618, 7.9709, 6.1632, 4.7546, 4.1182, 3.8926]), (X1_SIZE, 1))
A2_grid = np.tile(np.array([0.001, 12.9696, 159.9975, 48.7086, 20.6098, 12.1711, 8.0068, 4.448, 4.2918, 3.8111]), (X1_SIZE, 1))
k_grid = np.tile(np.array([1.235, 1.019, 0.693, 0.528, 0.417, 0.347, 0.3, 0.259, 0.231, 0.209]), (X1_SIZE, 1))
X0_GENOME = np.concatenate([A1_grid.flatten(), A2_grid.flatten(), k_grid.flatten(), Td_tile])

ORIGINAL_UPPER_BOUND = [200.0] * X1_SIZE * U_SIZE * 2 + [2.0] * X1_SIZE * U_SIZE + [5.0] * U_SIZE
SMALL_UPPER_BOUND = [200.0] * SMALL_X1_SIZE * SMALL_U_SIZE * 2 + [2.0] * SMALL_X1_SIZE * SMALL_U_SIZE + [5.0] * SMALL_U_SIZE
BOUNDS = np.array([(0.0,1.0) * (X1_SIZE * U_SIZE * 3 + U_SIZE)]).reshape((X1_SIZE * U_SIZE * 3 + U_SIZE),2)
SMALL_BOUNDS = np.array([(0.0,1.0) * (SMALL_X1_SIZE * SMALL_U_SIZE * 3 + SMALL_U_SIZE)]).reshape((SMALL_X1_SIZE * SMALL_U_SIZE * 3 + SMALL_U_SIZE),2)

EVALUATION_LIMIT = 10000
REPEATS = 10

@dataclass
class OptimizeResult:
    x: np.ndarray
    nfev: int
    fun: float
    nit: int

high_flexibility_costfun = prepare_stopping_costfun(TS, (X1_SIZE, U_SIZE), ((0.0, 20.0), (10.0, 100.0)), YMS)
low_flexibility_costfun = prepare_stopping_costfun(TS, (SMALL_X1_SIZE, SMALL_U_SIZE), ((0.0, 20.0), (10.0, 100.0)), YMS)
# high_flexibility_costfun = lambda x: np.sum(x**2)
# low_flexibility_costfun = lambda x: np.sum(x**2)

high_flexibility_problem = FunctionProblem(cma.ScaleCoordinates(high_flexibility_costfun, ORIGINAL_UPPER_BOUND), maximize=False, bounds=BOUNDS)
low_flexibility_problem = FunctionProblem(cma.ScaleCoordinates(low_flexibility_costfun, SMALL_UPPER_BOUND), maximize=False, bounds=SMALL_BOUNDS)

def prepare_two_level_hms_config():
    gsc = FitnessEvalLimitReached(EVALUATION_LIMIT)
    wrapped_root_function_problem = EvalCutoffProblem(low_flexibility_problem, eval_cutoff=EVALUATION_LIMIT)
    wrapped_leaf_function_problem = EvalCutoffProblem(high_flexibility_problem, eval_cutoff=EVALUATION_LIMIT)

    level_config = [
        DELevelConfig(
                generations=2,
                problem=wrapped_root_function_problem,
                pop_size=60,
                dither=True,
                crossover=0.9,
                lsc=DontStop(),
            ),
        CMALevelConfig(
            generations=20,
            problem=wrapped_leaf_function_problem,
            sigma0=0.05,
            lsc=FitnessSteadiness(max_deviation=0.005, n_metaepochs=5)
        ),
    ]
    sprout_condition = get_NBC_sprout(gen_dist_factor=2.25, trunc_factor=0.8, fil_dist_factor=2.25, level_limit=5)
    return TreeConfig(level_config, gsc, sprout_condition, options={"separate_costfuns": True})

def prepare_hms_config():
    gsc = SingularProblemEvalLimitReached(EVALUATION_LIMIT)
    wrapped_function_problem = EvalCutoffProblem(high_flexibility_problem, eval_cutoff=EVALUATION_LIMIT)
    
    level_config = [
        DELevelConfig(
                generations=2,
                problem=wrapped_function_problem,
                pop_size=100,
                dither=True,
                crossover=0.9,
                lsc=DontStop(),
            ),
        CMALevelConfig(
            generations=20,
            problem=wrapped_function_problem,
            sigma0=0.05,
            lsc=FitnessSteadiness(max_deviation=0.005, n_metaepochs=5)
        ),
    ]
    sprout_condition = get_NBC_sprout(gen_dist_factor=2.25, trunc_factor=0.8, fil_dist_factor=2.25, level_limit=5)
    return TreeConfig(level_config, gsc, sprout_condition, options={"hibernation": False})

def prepare_shade_config():
    gsc = SingularProblemEvalLimitReached(EVALUATION_LIMIT)
    wrapped_function_problem = EvalCutoffProblem(high_flexibility_problem, eval_cutoff=EVALUATION_LIMIT)

    level_config = [
        SHADELevelConfig(
            pop_size=200,
            problem=wrapped_function_problem,
            lsc=FitnessSteadiness(max_deviation=0.001, n_metaepochs=10),
            memory_size=100,
            generations=50, # We are working with one-level algorithm we do not want sprout mechanism to trigger needlesly yet we want to check for gsc
        ),
    ]
    sprout_condition = get_NBC_sprout()
    return TreeConfig(level_config, gsc, sprout_condition, options={"hibernation": False})

def prepare_cma_config():
    gsc = SingularProblemEvalLimitReached(EVALUATION_LIMIT)
    wrapped_function_problem = EvalCutoffProblem(high_flexibility_problem, eval_cutoff=EVALUATION_LIMIT)

    level_config = [
        CMALevelConfig(
            generations=50, # We are working with one-level algorithm we do not want sprout mechanism to trigger needlesly yet we want to check for gsc
            problem=wrapped_function_problem,
            sigma0=0.05,
            lsc=FitnessSteadiness(max_deviation=0.001, n_metaepochs=10),
            cma_options = {'CMA_elitist': "initial", "popsize_factor": 2}
        ),
    ]
    sprout_condition = get_NBC_sprout()
    return TreeConfig(level_config, gsc, sprout_condition, options={"hibernation": False})

def singular_run(tree_config: TreeConfig, random_seed: int):
    tree_config.options.update({"random_seed": random_seed, "log_level": LoggingLevel.WARNING})
    hms_tree = DemeTree(tree_config)
    hms_tree.run()
    hms_tree.pickle_dump(f"res/hms_results_{random_seed}.pkl")
    with open("res/results_backup.csv", "a") as f:
        csv.writer(f).writerow([random_seed, hms_tree.best_individual.genome, hms_tree.best_individual.fitness, hms_tree.n_evaluations, hms_tree.metaepoch_count])

if __name__ == '__main__':
    run_args = [prepare_two_level_hms_config()]*REPEATS + [prepare_hms_config()]*REPEATS + [prepare_shade_config()]*REPEATS + [prepare_cma_config()]*REPEATS
    run_args = list(zip(run_args, list(range(len(run_args)))))
    with Pool(8) as p:
        p.starmap(singular_run, run_args)
    