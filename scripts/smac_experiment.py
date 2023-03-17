import csv

from multiprocess import Pool
from smac_functions import run_smac_experiment

if __name__ == '__main__':

    max_pool = 55

    evaluation_limits = [10000]
    hms_level_configurations = ["sea_2_grad", "sea_2_no_grad", "cma_3"]
    func_sets = range(0, 5)

    parameters = []
    for i in range(5):
        for evaluation_limit in evaluation_limits:
            for level_configuration in hms_level_configurations:
                for func_set in func_sets:
                    parameters.append({'eval': evaluation_limit, 'config': level_configuration, 'test_problem': func_set, 'run_minimize': True})

    header = ['HMS variant', 'evaluation factor', 'test suite', 'config']
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)

    with Pool(max_pool) as p:
        pool_outputs = p.map(run_smac_experiment, parameters)

    header = ['HMS variant', 'evaluation factor', 'test suite', 'config']
    with open('smac_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
    
    for result in pool_outputs:
        with open('smac_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(result)
