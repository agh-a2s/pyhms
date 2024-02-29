import copy
import csv

from cma.bbobbenchmarks import instantiate

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pyhms.hms import hms
from pyhms.tree import DemeTree
from pyhms.config import EALevelConfig, CMALevelConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.problem import EvalCutoffProblem, FunctionProblem
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.sprout.sprout_filters import NBC_FarEnough, DemeLimit, LevelLimit
from pyhms.sprout.sprout_generators import NBC_Generator
from pyhms.stop_conditions.gsc import singular_problem_eval_limit_reached
from pyhms.stop_conditions.usc import metaepoch_limit, dont_stop

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario


def get_historic_best(tree: DemeTree) -> float:
    minimum = 2147483647.0

    for _, deme in tree.all_demes:
        for pop in deme.history:
            for ind in pop:
                if ind.fitness < minimum:
                    minimum = ind.fitness
    return minimum

def set1_separable(dim):
    sphere, op1 = instantiate(1)
    ellipsoidal, op2 = instantiate(2)
    rastrigin, op3 = instantiate(3)
    other_rastrigin, op4 = instantiate(4)
    linear_slope, op5 = instantiate(5)
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: sphere(x), maximize=False), dim*950, op1),
            EvalCutoffProblem(FunctionProblem(lambda x: ellipsoidal(x), maximize=False), dim*950, op2),
            EvalCutoffProblem(FunctionProblem(lambda x: rastrigin(x), maximize=False), dim*950, op3),
            EvalCutoffProblem(FunctionProblem(lambda x: other_rastrigin(x), maximize=False), dim*950, op4),
            EvalCutoffProblem(FunctionProblem(lambda x: linear_slope(x), maximize=False), dim*950, op5)]

def set2_low_conditioning(dim):
    attractive_sector, op1 = instantiate(6)
    step_ellipsoidal, op2 = instantiate(7)
    rosenbrock, op3 = instantiate(8)
    rosenbrock_rotated, op4 = instantiate(9)
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: attractive_sector(x), maximize=False), dim*950, op1),
            EvalCutoffProblem(FunctionProblem(lambda x: step_ellipsoidal(x), maximize=False), dim*950, op2),
            EvalCutoffProblem(FunctionProblem(lambda x: rosenbrock(x), maximize=False), dim*950, op3),
            EvalCutoffProblem(FunctionProblem(lambda x: rosenbrock_rotated(x), maximize=False), dim*950, op4)]

def set3_high_conditioning(dim):
    other_ellipsoidal, op1 = instantiate(10)
    discus, op2 = instantiate(11)
    bent_cigar, op3 = instantiate(12)
    sharp_ridge, op4 = instantiate(13)
    diff_powers, op5 = instantiate(14)
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: other_ellipsoidal(x), maximize=False), dim*950, op1),
            EvalCutoffProblem(FunctionProblem(lambda x: discus(x), maximize=False), dim*950, op2),
            EvalCutoffProblem(FunctionProblem(lambda x: bent_cigar(x), maximize=False), dim*950, op3),
            EvalCutoffProblem(FunctionProblem(lambda x: sharp_ridge(x), maximize=False), dim*950, op4),
            EvalCutoffProblem(FunctionProblem(lambda x: diff_powers(x), maximize=False), dim*950, op5)]

def set4_mmod_strong_struct(dim):
    rastrigin, op1 = instantiate(15)
    weierstrass, op2 = instantiate(16)
    schaffers_f7, op3 = instantiate(17)
    schaffers_f7_ill_cond, op4 = instantiate(18)
    griewank_rosenbrock, op5 = instantiate(19)
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: rastrigin(x), maximize=False), dim*950, op1),
            EvalCutoffProblem(FunctionProblem(lambda x: weierstrass(x), maximize=False), dim*950, op2),
            EvalCutoffProblem(FunctionProblem(lambda x: schaffers_f7(x), maximize=False), dim*950, op3),
            EvalCutoffProblem(FunctionProblem(lambda x: schaffers_f7_ill_cond(x), maximize=False), dim*950, op4),
            EvalCutoffProblem(FunctionProblem(lambda x: griewank_rosenbrock(x), maximize=False), dim*950, op5)]

def set5_mmod_weak_struct(dim):
    schwefel, op1 = instantiate(20)
    gallaghers_101_peaks, op2 = instantiate(21)
    gallaghers_21_peaks, op3 = instantiate(22)
    katsuura, op4 = instantiate(23)
    lunacek_bi_rastrigin, op5 = instantiate(24)
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: schwefel(x), maximize=False), dim*950, op1),
            EvalCutoffProblem(FunctionProblem(lambda x: gallaghers_101_peaks(x), maximize=False), dim*950, op2),
            EvalCutoffProblem(FunctionProblem(lambda x: gallaghers_21_peaks(x), maximize=False), dim*950, op3),
            EvalCutoffProblem(FunctionProblem(lambda x: katsuura(x), maximize=False), dim*950, op4),
            EvalCutoffProblem(FunctionProblem(lambda x: lunacek_bi_rastrigin(x), maximize=False), dim*950, op5)]


def prepare_evaluator_sea_2(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_sea_2(config, seed: int = 0):
        gsc = singular_problem_eval_limit_reached(limit=evaluation_factor)
        sprout = SproutMechanism(NBC_Generator(config["nbc_cut"], config["nbc_trunc"]), 
                                 [NBC_FarEnough(config["nbc_far"], 2), DemeLimit(1)], 
                                 [LevelLimit(config["level_limit"])])
        options = {'hibernation': False}

        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem)
            config_sea2 = [
            EALevelConfig(
                ea_class=SEA, 
                generations=config["generations1"],
                problem=problem, 
                bounds=bounds*dim_number, 
                pop_size=config["pop1"],
                mutation_std=config["mutation1"],
                lsc=dont_stop()
                ),
            EALevelConfig(
                ea_class=SEA, 
                generations=config["generations2"], 
                problem=problem, 
                bounds=bounds*dim_number, 
                pop_size=config["pop2"],
                mutation_std=config["mutation2"],
                sample_std_dev=config["sample_dev2"],
                lsc=metaepoch_limit(config["meataepoch2"]),
                run_minimize=True
                )
            ]
            tree = hms(level_config=config_sea2, gsc=gsc, sprout_cond=sprout, options=options)
            tree.run()
            if len(tree.optima) > 0:
                minimas.append(get_historic_best(tree))
            else:
                return 2147483647.0
        return sum(minimas)
    return evaluator_sea_2

def prepare_evaluator_cma_2(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_cma_2(config, seed: int = 0):
        gsc = singular_problem_eval_limit_reached(limit=evaluation_factor)
        sprout = SproutMechanism(NBC_Generator(config["nbc_cut"], config["nbc_trunc"]), 
                                 [NBC_FarEnough(config["nbc_far"], 2), DemeLimit(1)], 
                                 [LevelLimit(config["level_limit"])])
        options = {'hibernation': False}

        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem)
            config_cma2 = [
            EALevelConfig(
                ea_class=SEA, 
                generations=config["generations1"], 
                problem=problem, 
                bounds=bounds*dim_number, 
                pop_size=config["pop1"],
                mutation_std=config["mutation1"],
                lsc=dont_stop()
                ),
            CMALevelConfig(
                problem=problem, 
                bounds=bounds*dim_number,
                lsc=metaepoch_limit(config["meataepoch2"]),
                sigma0=config["sigma2"],
                generations=config["generations2"]
                )
            ]
            tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout, options=options)
            tree.run()
            if len(tree.optima) > 0:
                minimas.append(get_historic_best(tree))
            else:
                return 2147483647.0
        return sum(minimas)
    return evaluator_cma_2

def prepare_sea_2(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_cut", 1.5, 4.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_trunc", 0.1, 0.9))
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_far", 1.5, 4.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop2", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations1", 1, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations2", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 5, 40))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation2", 0.01, 1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sample_dev2", 0.1, 3.0))
    
    # Meta data for the optimization
    scenario = Scenario(configspace, deterministic=False, n_trials=3000)
    
    evaluator = prepare_evaluator_sea_2(test_problem, dim_number, bounds, evaluation_factor)

    return HyperparameterOptimizationFacade(scenario, evaluator)

def prepare_cma_2(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_cut", 1.5, 4.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_trunc", 0.1, 0.9))
    configspace.add_hyperparameter(UniformFloatHyperparameter("nbc_far", 1.5, 4.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations1", 1, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations2", 3, 30))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 30, 300))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma2", 0.1, 3.0))

    # Meta data for the optimization
    scenario = Scenario(configspace, deterministic=False, n_trials=3000)
    
    evaluator = prepare_evaluator_cma_2(test_problem, dim_number, bounds, evaluation_factor)

    return HyperparameterOptimizationFacade(scenario, evaluator)

def prepare_config(hms_variant, test_problem, dim_number, bounds, evaluation_factor):
    if hms_variant == "sea_2":
        return prepare_sea_2(test_problem, dim_number, bounds, evaluation_factor)
    elif hms_variant == "cma_2":
        return prepare_cma_2(test_problem, dim_number, bounds, evaluation_factor)

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
