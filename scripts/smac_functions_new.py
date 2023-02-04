import ast
import copy
import csv
import numpy as np
import pandas as pd
import pickle as pkl

from pyhms.pyhms.hms import hms
from pyhms.pyhms.config import EALevelConfig, CMALevelConfig
from pyhms.pyhms.demes.single_pop_eas.sea import SEA
from pyhms.pyhms.problem import EvalCountingProblem, FunctionProblem
from pyhms.pyhms.sprout import composite_condition, far_enough, level_limit
from pyhms.pyhms.stop_conditions.gsc import fitness_eval_limit_reached
from pyhms.pyhms.stop_conditions.usc import metaepoch_limit, dont_stop

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import BoolVector, FloatVector, FloatMatrix, IntVector, StrVector
from rpy2.robjects import ListVector, NA_Complex, NA_Character, NA_Integer
import rpy2.robjects.packages as rpackages
import rpy2.robjects as R
from rpy2.robjects import numpy2ri

base = importr('base')
utils = importr('utils')
smoof = importr('smoof', lib_loc='~/R_libs/')

# utils.chooseCRANmirror(ind=1)
# packnames = ['smoof']
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install), lib = "/usr/lib/R/x86_64-pc-linux-gnu-library/3.6")
# smoof = importr('smoof')

def prepare_evaluator_sea_2(test_problem, dim_number, bounds, evaluation_factor):
    def evaluator_sea_2(config):
        gsc = fitness_eval_limit_reached(limit=dim_number*evaluation_factor, weights=None)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), level_limit(config["level_limit"])])

        config_sea2 = [
        EALevelConfig(
            ea_class=SEA, 
            generations=config["generations"],
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number, 
            pop_size=config["pop1"],
            mutation_std=config["mutation1"],
            lsc=dont_stop()
            ),
        EALevelConfig(
            ea_class=SEA, 
            generations=config["generations"], 
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number, 
            pop_size=config["pop2"],
            mutation_std=config["mutation2"],
            sample_std_dev=config["sample_dev2"],
            lsc=metaepoch_limit(config["meataepoch2"])
            )
        ]
        tree = hms(level_config=config_sea2, gsc=gsc, sprout_cond=sprout_cond)
        if len(tree.optima) > 0:
            return min([o.fitness for o in tree.optima])
        else:
            return 2147483647.0
    return evaluator_sea_2

def prepare_evaluator_cma_2(test_problem, dim_number, bounds, evaluation_factor):
    def evaluator_cma_2(config):
        gsc = fitness_eval_limit_reached(limit=dim_number*evaluation_factor, weights=None)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), level_limit(config["level_limit"])])

        config_cma2 = [
        EALevelConfig(
            ea_class=SEA, 
            generations=config["generations"], 
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number, 
            pop_size=config["pop1"],
            mutation_std=config["mutation1"],
            lsc=dont_stop()
            ),
        CMALevelConfig(
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number,
            lsc=metaepoch_limit(config["meataepoch2"]),
            sigma0=config["sigma2"],
            generations=config["generations"]
            )
        ]
        tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond)
        if len(tree.optima) > 0:
            return min([o.fitness for o in tree.optima])
        else:
            return 2147483647.0
    return evaluator_cma_2

def prepare_evaluator_cma_3(test_problem, dim_number, bounds, evaluation_factor):
    def evaluator_cma_3(config):
        gsc = fitness_eval_limit_reached(limit=dim_number*evaluation_factor, weights=None)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), level_limit(config["level_limit"])])

        config_cma3 = [
        EALevelConfig(
            ea_class=SEA, 
            generations=config["generations"], 
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number, 
            pop_size=config["pop1"],
            mutation_std=config["mutation1"],
            lsc=dont_stop()
            ),
        EALevelConfig(
            ea_class=SEA, 
            generations=config["generations"], 
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number, 
            pop_size=config["pop2"],
            mutation_std=config["mutation2"],
            sample_std_dev=config["sample_dev2"],
            lsc=metaepoch_limit(config["meataepoch2"])
            ),
        CMALevelConfig(
            problem=copy.deepcopy(test_problem), 
            bounds=bounds*dim_number,
            lsc=metaepoch_limit(config["meataepoch3"]),
            sigma0=config["sigma3"],
            generations=config["generations"]
            )
        ]
        tree = hms(level_config=config_cma3, gsc=gsc, sprout_cond=sprout_cond)
        if len(tree.optima) > 0:
            return min([o.fitness for o in tree.optima])
        else:
            return 2147483647.0
    return evaluator_cma_3
    
def prepare_sea_2(function_set, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.25, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 50, 500))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop2", 50, 500))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 5, 40))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation2", 0.01, 1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sample_dev2", 0.1, 3.0))
    
    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 3,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 1 time with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_sea_2(function_set, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_cma_2(function_set, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.25, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 50, 500))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 50, 400))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma2", 0.01, 2.0))

    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 3,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 1 time with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_cma_2(function_set, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_cma_3(function_set, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.25, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 50, 500))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop2", 50, 500))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 5, 20))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch3", 50, 400))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation2", 0.01, 1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sample_dev2", 0.1, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma3", 0.01, 2.0))

    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 3,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 1 time with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_cma_3(function_set, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_config(hms_variant, function_set, dim_number, bounds, evaluation_factor):
    if hms_variant == "sea_2":
        return prepare_sea_2(function_set(dim_number), dim_number, bounds, evaluation_factor)
    elif hms_variant == "cma_2":
        return prepare_cma_2(function_set(dim_number), dim_number, bounds, evaluation_factor)
    elif hms_variant == "cma_3":
        return prepare_cma_3(function_set(dim_number), dim_number, bounds, evaluation_factor)

def run_smac_experiment(parameters):
    bbob_fun = smoof.makeBBOBFunction(parameters['dim'], parameters['test_problem'], 1)
    test_problem = EvalCountingProblem(FunctionProblem(lambda x: bbob_fun(FloatVector(x))[0], maximize=False))
    experiment = prepare_config(parameters['config'], test_problem, parameters['dim'], parameters['bounds'], parameters['eval'], True)
    best_found_config = experiment.optimize()
    rh = experiment.get_runhistory()
    fit = rh.get_instance_costs_for_config(best_found_config)
    result = [parameters['config'], parameters['eval'], parameters['dim'], parameters['test_problem'], best_found_config.get_dictionary(), list(fit.values())[0]]
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(result)
    return result
