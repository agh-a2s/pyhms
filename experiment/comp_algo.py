import copy
import csv
import numpy as np

import ast
import copy
import csv
import random
import pickle as pkl
from multiprocess import Pool

import pyade.ilshade
from cma import fmin
from cma.bbobbenchmarks import instantiate

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def evaluator_ilshade(config, eval_limit, dim_number, function_problem, random_state, bounds):
    problem=copy.deepcopy(function_problem)

    algorithm = pyade.ilshade

    params = algorithm.get_default_params(dim=dim_number)
    params['bounds'] = np.array(bounds*dim_number)
    params['population_size'] = config['popsize']
    params['memory_size'] = config['archivesize']
    params['func'] = problem
    params['max_evals'] = eval_limit

    np.random.seed(random_state)
    random.seed(random_state)
    _, fitness = algorithm.apply(**params)
    return fitness

def evaluator_cmaes(config, eval_limit, dim_number, function_problem, random_state, bounds):
    problem=copy.deepcopy(function_problem)
    cma_lb = [bounds[0][0]]*dim_number
    cma_ub = [bounds[0][1]]*dim_number
    start = np.random.uniform(bounds[0][0], bounds[0][1], size=dim_number)
    sigma0 = config['sigma0']
    popsize = config['popsize']
    incpopsize = config['incpopsize']
    tolfun = config['tolfun']
    options = {'bounds': [cma_lb, cma_ub], 'verbose': -9, 'maxfevals': eval_limit, 'popsize': popsize, 'tolfun': tolfun, 'seed':random_state, 'randn':np.random.randn}

    res = fmin(problem, start, sigma0, options=options, bipop=True, restart_from_best=True, restarts=10, incpopsize=incpopsize)
    return res[1]

if __name__ == '__main__':

    max_pool = 12

    dim_number = [10,20]
    bounds = [(-5,5)]
    evaluation_factor = 1000
    ela_evaluation_factor = 50
    variants = {'ilshade': evaluator_ilshade, 'cmaes': evaluator_cmaes}
    test_sets = {0: [1,2,3,4,5], 1: [6,7,8,9], 2: [10,11,12,13,14], 3: [15,16,17,18,19], 4: [20,21,22,23,24]}

    with open("results/comp_results.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        counter = 0
        header = []
        hms_configs = []
        next(reader, None)
        for row in reader:
            hms_configs.append([row[0], int(float(row[1])), int(float(row[2])), int(float(row[3])), ast.literal_eval(row[4])])

    configs = []
    for dim in dim_number:
        for i in range(0, len(hms_configs)):
            configs.append((dim, hms_configs[i][3], hms_configs[i]))

    def test_config(config):
        bounds = [(-5,5)]
        evaluation_factor = 1000
        ela_evaluation_factor = 50
        variants = {'ilshade': evaluator_ilshade, 'cmaes': evaluator_cmaes}
        test_sets = {0: [1,2,3,4,5], 1: [6,7,8,9], 2: [10,11,12,13,14], 3: [15,16,17,18,19], 4: [20,21,22,23,24]}
        dim = config[0]
        test_problem = test_sets[config[1]]
        algo_config = config[2]
        evaluator = variants[algo_config[0]]
        result = []
        for j in test_sets[algo_config[3]]:
            bbob_fun, opt = instantiate(j)
            result_tmp = []
            for k in range(100):
                test_problem = bbob_fun
                fit = evaluator(algo_config[4], (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem, k+1, bounds)
                result_tmp.append(fit - opt)
            result.append((result_tmp, j))
            with open('backup_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow((result_tmp, dim, j, algo_config[4], np.mean(result_tmp)))
                print((dim, j, np.mean(result_tmp), algo_config[4]))
        return (result, dim, algo_config[4])

    with Pool(max_pool) as p:
        param_results = p.map(test_config, configs)

    with open('algo_conf_evaluated.pkl', 'wb') as pickle_file:
        pkl.dump(param_results, pickle_file)