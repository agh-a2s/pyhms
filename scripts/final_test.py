import copy
import csv
import numpy as np
import pickle as pkl
import random
random.seed(1)
np.random.seed(1)
from pyhms.pyhms.hms import hms
from pyhms.pyhms.config import EALevelConfig, CMALevelConfig
from pyhms.pyhms.demes.single_pop_eas.sea import SEA
from pyhms.pyhms.problem import EvalCutoffProblem, FunctionProblem
from pyhms.pyhms.sprout import composite_condition, far_enough, deme_per_level_limit
from pyhms.pyhms.stop_conditions.gsc import fitness_eval_limit_reached
from pyhms.pyhms.stop_conditions.usc import metaepoch_limit, dont_stop
from cma.bbobbenchmarks import instantiate
from multiprocess import Pool

def evaluator_sea_2(config, eval_limit, dim_number, function_problem):
    gsc = fitness_eval_limit_reached(limit=eval_limit, weights=None)
    sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])

    config_sea2 = [
    EALevelConfig(
        ea_class=SEA, 
        generations=config["generations1"],
        problem=copy.deepcopy(function_problem), 
        bounds=bounds*dim_number, 
        pop_size=config["pop1"],
        mutation_std=config["mutation1"],
        lsc=dont_stop()
        ),
    EALevelConfig(
        ea_class=SEA, 
        generations=config["generations2"], 
        problem=copy.deepcopy(function_problem), 
        bounds=bounds*dim_number, 
        pop_size=config["pop2"],
        mutation_std=config["mutation2"],
        sample_std_dev=config["sample_dev2"],
        lsc=metaepoch_limit(config["meataepoch2"]),
        run_minimize=True
        )
    ]
    tree = hms(level_config=config_sea2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

def evaluator_cma_2(config, eval_limit, dim_number, function_problem):
    gsc = fitness_eval_limit_reached(limit=eval_limit, weights=None)
    sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])

    config_cma2 = [
    EALevelConfig(
        ea_class=SEA, 
        generations=config["generations1"], 
        problem=copy.deepcopy(function_problem), 
        bounds=bounds*dim_number, 
        pop_size=config["pop1"],
        mutation_std=config["mutation1"],
        lsc=dont_stop()
        ),
    CMALevelConfig(
        problem=copy.deepcopy(function_problem), 
        bounds=bounds*dim_number,
        lsc=metaepoch_limit(config["meataepoch2"]),
        sigma0=config["sigma2"],
        generations=config["generations2"]
        )
    ]
    tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

if __name__ == '__main__':
    final_configs = {1: {'far_enough': 0.288769018813231,
        'generations1': 2,
        'generations2': 8,
        'level_limit': 3,
        'meataepoch2': 185,
        'mutation1': 2.7548740563169476,
        'pop1': 22,
        'sigma2': 2.8308793657377334},
        2: {'far_enough': 5.533764915621696,
        'generations1': 3,
        'generations2': 2,
        'level_limit': 10,
        'meataepoch2': 12,
        'mutation1': 0.30113932430261714,
        'mutation2': 0.03637808039096236,
        'pop1': 29,
        'pop2': 20,
        'sample_dev2': 0.9375043636862485},
        3: {'far_enough': 18.124365539946634,
        'generations1': 2,
        'generations2': 10,
        'level_limit': 6,
        'meataepoch2': 139,
        'mutation1': 0.41367275532017866,
        'pop1': 21,
        'sigma2': 0.16704630552698663},
        4: {'far_enough': 13.850188638220365,
        'generations1': 2,
        'generations2': 9,
        'level_limit': 5,
        'meataepoch2': 86,
        'mutation1': 1.265542040510019,
        'pop1': 25,
        'sigma2': 1.2710313593749352},
        5: {'far_enough': 16.85391591224264,
        'generations1': 2,
        'generations2': 6,
        'level_limit': 4,
        'meataepoch2': 6,
        'mutation1': 2.5459184638416805,
        'mutation2': 0.2906046628084067,
        'pop1': 78,
        'pop2': 35,
        'sample_dev2': 0.9618745655809818}}

    dim_number = [10,20]
    bounds = [(-5,5)]
    evaluation_factor = 1000
    ela_evaluation_factor = 50
    test_sets = {1: [1,2,3,4,5], 2: [6,7,8,9], 3: [10,11,12,13,14], 4: [15,16,17,18,19], 5: [20,21,22,23,24]}

    max_pool = 48
    test_configs_unfolded = []

    for dim in dim_number:
        for config in final_configs.values():
            for test_set in test_sets.values():
                for test_problem in test_set:
                    test_configs_unfolded.append((dim, test_problem, config))

    def test_config(test_config):
        dim = test_config[0]
        bbob_fun = instantiate(test_config[1])[0]
        hms_config = test_config[2]

        result = []
        for _ in range(100):
            test_problem = EvalCutoffProblem(FunctionProblem(bbob_fun, maximize=False), (evaluation_factor-ela_evaluation_factor)*dim)
            if "sigma2" in hms_config:
                tree = evaluator_cma_2(hms_config, (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
            else:
                tree = evaluator_sea_2(hms_config, (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
            fit = 2147483647.0
            for level_no in range(tree.height):
                for deme in tree.levels[level_no]:
                    for pop in deme.history:
                        for ind in pop:
                            if ind.fitness < fit:
                                fit = ind.fitness
                result.append(fit)
        with open('final_results_0.csv', 'a') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow((result, dim, test_config[1], hms_config))
        return (result, dim, test_config[1], hms_config)
    
    with Pool(max_pool) as p:
        hms_param_results = p.map(test_config, test_configs_unfolded)

    for result in hms_param_results:
        with open('final_results_0.csv', 'a') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(result)