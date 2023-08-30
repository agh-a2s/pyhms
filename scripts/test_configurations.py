import ast
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
from pyhms.pyhms.stop_conditions.gsc import singular_problem_eval_limit_reached
from pyhms.pyhms.stop_conditions.usc import metaepoch_limit, dont_stop
from cma.bbobbenchmarks import instantiate
from multiprocess import Pool

def evaluator_sea_2(config, eval_limit, dim_number, function_problem):
    gsc = singular_problem_eval_limit_reached(limit=eval_limit)
    sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
    problem=copy.deepcopy(function_problem)

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
    tree = hms(level_config=config_sea2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

def evaluator_cma_2(config, eval_limit, dim_number, function_problem):
    gsc = singular_problem_eval_limit_reached(limit=eval_limit)
    sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
    problem=copy.deepcopy(function_problem)

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
    tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

def evaluator_cma_3(config, eval_limit, dim_number, function_problem):
    gsc = singular_problem_eval_limit_reached(limit=eval_limit)
    sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
    problem=copy.deepcopy(function_problem)

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
    EALevelConfig(
            ea_class=SEA, 
            generations=config["generations2"], 
            problem=problem, 
            bounds=bounds*dim_number, 
            pop_size=config["pop2"],
            mutation_std=config["mutation2"],
            sample_std_dev=config["sample_dev2"],
            lsc=metaepoch_limit(config["meataepoch2"])
            ),
    CMALevelConfig(
        problem=problem, 
        bounds=bounds*dim_number,
        lsc=metaepoch_limit(config["meataepoch3"]),
        sigma0=config["sigma3"],
        generations=config["generations3"]
        )
    ]
    tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

if __name__ == '__main__':

    benchmark = []
    dim_number = [10,20]
    bounds = [(-5,5)]
    evaluation_factor = 1000
    ela_evaluation_factor = 50
    variants = {'sea_2': evaluator_sea_2, 'cma_2': evaluator_cma_2, 'cma_3': evaluator_cma_3}
    test_sets = {0: [1,2,3,4,5], 1: [6,7,8,9], 2: [10,11,12,13,14], 3: [15,16,17,18,19], 4: [20,21,22,23,24]}

    with open("smac_results.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        counter = 0
        header = []
        hms_configs = []
        next(reader, None)
        for row in reader:
            hms_configs.append([row[0], int(float(row[1])), int(float(row[2])), int(float(row[3])), ast.literal_eval(row[4])])


    max_pool = 55

    configs = []
    for dim in dim_number:
        for i in range(0, len(hms_configs)):
            configs.append((dim, hms_configs[i][3], hms_configs[i]))

    def test_config(config):
        dim = config[0]
        test_problem = test_sets[config[1]]
        hms_config = config[2]
        evaluator = variants[hms_config[0]]
        result = []
        for j in test_sets[hms_config[3]]:
            bbob_fun = instantiate(j)[0]
            result_tmp = []
            for _ in range(50):
                test_problem = EvalCutoffProblem(FunctionProblem(bbob_fun, maximize=False), (evaluation_factor-ela_evaluation_factor)*dim)
                tree = evaluator(hms_config[4], (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
                if len([o.fitness for o in tree.optima]) > 0:
                    fit = tree.historic_best.fitness
                else:
                    fit = 2147483647.0
                result_tmp.append(fit)
            result.append((result_tmp, j))
            with open('backup_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow((result_tmp, dim, j, hms_config[4]))
        return (result, dim, hms_config[4])
    
    def test_config_on_all(config):
        dim = config[0]
        test_problem = test_sets[config[1]]
        hms_config = config[2]
        evaluator = variants[hms_config[0]]
        result = []
        for j in range(1, 25):
            bbob_fun = instantiate(j)[0]
            result_tmp = []
            for _ in range(50):
                test_problem = EvalCutoffProblem(FunctionProblem(bbob_fun, maximize=False), (evaluation_factor-ela_evaluation_factor)*dim)
                tree = evaluator(hms_config[4], (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
                if len([o.fitness for o in tree.optima]) > 0:
                    fit = tree.historic_best.fitness
                else:
                    fit = 2147483647.0
                result_tmp.append(fit)
            result.append((result_tmp, j))
            with open('backup_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow((result_tmp, dim, j, hms_config[4]))
        return (result, dim, hms_config[4])
    
    with Pool(max_pool) as p:
        # hms_param_results = p.map(test_config, configs)
        hms_param_results = p.map(test_config_on_all, configs)

    with open('hms_conf_evaluated.pickle', 'wb') as pickle_file:
        pkl.dump(hms_param_results, pickle_file)