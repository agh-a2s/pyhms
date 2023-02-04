import csv
import multiprocessing
import os

from multiprocess import Pool
from smac_functions_new import run_smac_experiment

if __name__ == '__main__':

    max_pool = 55

    evaluation_limits = [50000]
    dimensions_list = [10]
    bounds=[(-5, 5)]
    hms_level_configurations = ["sea_2", "cma_2", "cma_3"]
    func_sets = range(1, 25)

    parameters = []
    for i in range(5):
        for evaluation_limit in evaluation_limits:
            for dimensions in dimensions_list:
                for level_configuration in hms_level_configurations:
                    for func_set in func_sets:
                        parameters.append({'eval': evaluation_limit, 'dim': dimensions, 'config': level_configuration, 'test_problem': func_set, 'bounds': bounds})

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)

    with Pool(max_pool) as p:
        pool_outputs = list(
            p.imap(run_smac_experiment,
                    parameters),
            total=len(parameters)
        )

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('smac_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
    
    for result in pool_outputs:
        with open('smac_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(result)
