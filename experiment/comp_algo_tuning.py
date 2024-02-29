import copy
import csv
import numpy as np

import pyade.ilshade
from cma import fmin
from cma.bbobbenchmarks import instantiate

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario

def set1_separable(dim):
    sphere, op1 = instantiate(1)
    ellipsoidal, op2 = instantiate(2)
    rastrigin, op3 = instantiate(3)
    other_rastrigin, op4 = instantiate(4)
    linear_slope, op5 = instantiate(5)
    
    return [(lambda x: sphere(x), op1),
            (lambda x: ellipsoidal(x), op2),
            (lambda x: rastrigin(x), op3),
            (lambda x: other_rastrigin(x), op4),
            (lambda x: linear_slope(x), op5)]

def set2_low_conditioning(dim):
    attractive_sector, op1 = instantiate(6)
    step_ellipsoidal, op2 = instantiate(7)
    rosenbrock, op3 = instantiate(8)
    rosenbrock_rotated, op4 = instantiate(9)
    
    return [(lambda x: attractive_sector(x), op1),
            (lambda x: step_ellipsoidal(x), op2),
            (lambda x: rosenbrock(x), op3),
            (lambda x: rosenbrock_rotated(x), op4)]

def set3_high_conditioning(dim):
    other_ellipsoidal, op1 = instantiate(10)
    discus, op2 = instantiate(11)
    bent_cigar, op3 = instantiate(12)
    sharp_ridge, op4 = instantiate(13)
    diff_powers, op5 = instantiate(14)
    
    return [(lambda x: other_ellipsoidal(x), op1),
            (lambda x: discus(x), op2),
            (lambda x: bent_cigar(x), op3),
            (lambda x: sharp_ridge(x), op4),
            (lambda x: diff_powers(x), op5)]

def set4_mmod_strong_struct(dim):
    rastrigin, op1 = instantiate(15)
    weierstrass, op2 = instantiate(16)
    schaffers_f7, op3 = instantiate(17)
    schaffers_f7_ill_cond, op4 = instantiate(18)
    griewank_rosenbrock, op5 = instantiate(19)
    
    return [(lambda x: rastrigin(x), op1),
            (lambda x: weierstrass(x), op2),
            (lambda x: schaffers_f7(x), op3),
            (lambda x: schaffers_f7_ill_cond(x), op4),
            (lambda x: griewank_rosenbrock(x), op5)]

def set5_mmod_weak_struct(dim):
    schwefel, op1 = instantiate(20)
    gallaghers_101_peaks, op2 = instantiate(21)
    gallaghers_21_peaks, op3 = instantiate(22)
    katsuura, op4 = instantiate(23)
    lunacek_bi_rastrigin, op5 = instantiate(24)
    
    return [(lambda x: schwefel(x), op1),
            (lambda x: gallaghers_101_peaks(x), op2),
            (lambda x: gallaghers_21_peaks(x), op3),
            (lambda x: katsuura(x), op4),
            (lambda x: lunacek_bi_rastrigin(x), op5)]

def get_best(pop, fit):
    minima = 2147483647.0
    for i in range(len(pop)):
        if fit[i] < minima:
            minima = fit[i]
    return minima

def prepare_evaluator_ilshade(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_ilshade(config, seed: int = 0):
        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem[0])

            algorithm = pyade.ilshade

            params = algorithm.get_default_params(dim=dim_number)
            params['bounds'] = np.array(bounds*dim_number)
            params['population_size'] = config['popsize']
            params['memory_size'] = config['archivesize']
            params['func'] = problem
            params['max_evals'] = evaluation_factor

            _, fitness = algorithm.apply(**params)
            minimas.append(fitness)
        return sum(minimas)
    return evaluator_ilshade

def prepare_evaluator_cmaes(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_cmaes(config, seed: int = 0):
        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem[0])
            cma_lb = [bounds[0][0]]*dim_number
            cma_ub = [bounds[0][1]]*dim_number
            start = np.random.uniform(bounds[0][0], bounds[0][1], size=dim_number)
            sigma0 = config['sigma0']
            popsize = config['popsize']
            incpopsize = config['incpopsize']
            tolfun = config['tolfun']
            options = {'bounds': [cma_lb, cma_ub], 'verbose': -9, 'maxfevals': evaluation_factor, 'popsize': popsize, 'tolfun': tolfun}

            res = fmin(problem, start, sigma0, options=options, bipop=True, restart_from_best=True, restarts=10, incpopsize=incpopsize)
            minimas.append(res[1])
        return sum(minimas)
    return evaluator_cmaes

def prepare_ilshade(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformIntegerHyperparameter("popsize", 10, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("archivesize", 1, 20))
    
    # Meta data for the optimization
    scenario = Scenario(configspace, deterministic=False, n_trials=3000)
    
    evaluator = prepare_evaluator_ilshade(test_problem, dim_number, bounds, evaluation_factor)

    return HyperparameterOptimizationFacade(scenario, evaluator)

def prepare_cmaes(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformIntegerHyperparameter("popsize", 10, 100))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma0", 0.5, 7.5))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("incpopsize", 1, 5))
    configspace.add_hyperparameter(UniformFloatHyperparameter("tolfun", 1e-15, 1e-8))
    
    # Meta data for the optimization
    scenario = Scenario(configspace, deterministic=False, n_trials=3000)
    
    evaluator = prepare_evaluator_cmaes(test_problem, dim_number, bounds, evaluation_factor)

    return HyperparameterOptimizationFacade(scenario, evaluator)

def prepare_config(algo_variant, test_problem, dim_number, bounds, evaluation_factor):
    if algo_variant == "ilshade":
        return prepare_ilshade(test_problem, dim_number, bounds, evaluation_factor)
    elif algo_variant == "cmaes":
        return prepare_cmaes(test_problem, dim_number, bounds, evaluation_factor)

def run_smac_experiment(parameters):
    func_sets = [set1_separable, set2_low_conditioning, set3_high_conditioning, set4_mmod_strong_struct, set5_mmod_weak_struct]
    test_problem = func_sets[parameters['test_problem']]
    experiment = prepare_config(parameters['config'], test_problem(parameters['dim']), parameters['dim'], parameters['bounds'], parameters['eval'])
    best_found_config = experiment.optimize()
    result = [parameters['config'], parameters['eval'], parameters['dim'], parameters['test_problem'], best_found_config.get_dictionary()]
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(result)
    return result