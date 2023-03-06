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
from pyhms.pyhms.problem import EvalCountingProblem, FunctionProblem
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
        generations=config["generations"],
        problem=copy.deepcopy(function_problem), 
        bounds=bounds*dim_number, 
        pop_size=config["pop1"],
        mutation_std=config["mutation1"],
        lsc=dont_stop()
        ),
    EALevelConfig(
        ea_class=SEA, 
        generations=config["generations"], 
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
        generations=config["generations"], 
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
        generations=config["generations"]
        )
    ]
    tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond)
    return tree

def scaler(config, number_of_dimensions):
    if number_of_dimensions == 10:
        return config
    elif "sigma2" in config:
        config["pop1"] = int(config["pop1"] * number_of_dimensions/10)
        config["level_limit"] = int(config["level_limit"]*np.sqrt(number_of_dimensions/10))
        config["meataepoch2"] = int(config["meataepoch2"]*np.sqrt(number_of_dimensions/10))
    elif "sample_dev2" in config:
        config["pop1"] = int(config["pop1"] * number_of_dimensions/10)
        config["level_limit"] = int(config["level_limit"]*np.sqrt(number_of_dimensions/10))
        config["pop2"] = int(config["pop2"]*np.sqrt(number_of_dimensions/10))
    return config


if __name__ == '__main__':
    final_configs = {1: {'far_enough': 3.2767417667749554,'generations': 8,'level_limit': 2,'meataepoch2': 193,'mutation1': 1.456212193078683,'pop1': 20,'sigma2': 2.4817937329295887},
                2: {'far_enough': 1.321612313472642,'generations': 2,'level_limit': 2,'meataepoch2': 5,'mutation1': 0.3826177350881527,'mutation2': 0.12605311716085577,'pop1': 30,'pop2': 22,'sample_dev2': 1.4323836990304843},
                3: {'far_enough': 1.5046437629820386,'generations': 2,'level_limit': 2,'meataepoch2': 9,'mutation1': 0.8448276354043325,'mutation2': 0.4637405422423578,'pop1': 59,'pop2': 20,'sample_dev2': 1.9212872683468143},
                4: {'far_enough': 0.8714979335463446,'generations': 2,'level_limit': 3,'meataepoch2': 300,'mutation1': 0.33362686765472904,'pop1': 20,'sigma2': 1.7462625521061992},
                5: {'far_enough': 1.877044785767208,'generations': 4,'level_limit': 5,'meataepoch2': 9,'mutation1': 1.2891196372390588,'mutation2': 0.30970875374701495,'pop1': 27,'pop2': 29,'sample_dev2': 2.9749841623713347}}

    dim_number = [10,20]
    bounds = [(-5,5)]
    evaluation_factor = 1000
    ela_evaluation_factor = 50
    test_sets = {1: [1,2,3,4,5], 2: [6,7,8,9], 3: [10,11,12,13,14], 4: [15,16,17,18,19], 5: [20,21,22,23,24]}

    max_pool = 48

    test_configs = []
    test_configs.append((10, 1, ((49, scaler(final_configs[1], 10)), (1, scaler(final_configs[3], 10)))))
    test_configs.append((20, 1, ((49, scaler(final_configs[1], 20)), (1, scaler(final_configs[3], 20)))))
    test_configs.append((10, 2, ((45, scaler(final_configs[2], 10)), (3, scaler(final_configs[4], 10)), (1, scaler(final_configs[3], 10)), (1, scaler(final_configs[1], 10)))))
    test_configs.append((20, 2, ((45, scaler(final_configs[2], 20)), (3, scaler(final_configs[4], 20)), (1, scaler(final_configs[3], 20)), (1, scaler(final_configs[1], 20)))))
    test_configs.append((10, 3, ((43, scaler(final_configs[3], 10)), (2, scaler(final_configs[1], 10)), (2, scaler(final_configs[2], 10)), (3, scaler(final_configs[4], 10)))))
    test_configs.append((20, 3, ((43, scaler(final_configs[3], 20)), (2, scaler(final_configs[1], 20)), (2, scaler(final_configs[2], 20)), (3, scaler(final_configs[4], 20)))))
    test_configs.append((10, 4, ((43, scaler(final_configs[4], 10)), (1, scaler(final_configs[1], 10)), (2, scaler(final_configs[2], 10)), (2, scaler(final_configs[3], 10)), (2, scaler(final_configs[5], 10)))))
    test_configs.append((20, 4, ((43, scaler(final_configs[4], 20)), (1, scaler(final_configs[1], 20)), (2, scaler(final_configs[2], 20)), (2, scaler(final_configs[3], 20)), (2, scaler(final_configs[5], 20)))))
    test_configs.append((10, 5, ((49, scaler(final_configs[5], 10)), (1, scaler(final_configs[4], 10)))))
    test_configs.append((20, 5, ((49, scaler(final_configs[5], 20)), (1, scaler(final_configs[4], 20)))))

    test_configs_unfolded = []
    for config in test_configs:
        for i in test_sets[config[1]]:
            test_configs_unfolded.append((config[0], i, config[2]))

    def test_config(test_config):
        dim = test_config[0]
        bbob_fun = instantiate(test_config[1])[0]
        hms_configs = test_config[2]

        result = []
        for config in hms_configs:
            for _ in range(config[0]):
                test_problem = EvalCountingProblem(FunctionProblem(bbob_fun, maximize=False))
                if "sigma2" in config[1]:
                    tree = evaluator_cma_2(config[1], (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
                else:
                    tree = evaluator_sea_2(config[1], (evaluation_factor-ela_evaluation_factor)*dim, dim, test_problem)
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
            writer.writerow((result, dim, test_config[1]))
        return (result, dim, test_config[1])
    
    with Pool(max_pool) as p:
        hms_param_results = p.map(test_config, test_configs_unfolded)

    for result in hms_param_results:
        with open('final_results_0.csv', 'a') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(result)