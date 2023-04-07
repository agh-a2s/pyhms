import csv

from multiprocess import Pool
from smac_functions_precision import run_smac_experiment

if __name__ == '__main__':

    max_pool = 55

    dimensions_list = [10]
    bounds=[(-5, 5)]
    hms_level_configurations = ["sea_2", "cma_2"]
    func_sets = range(0, 5)

    parameters = []
    for i in range(5):
        for dimensions in dimensions_list:
            for level_configuration in hms_level_configurations:
                for func_set in func_sets:
                    parameters.append({'dim': dimensions, 'config': level_configuration, 'test_problem': func_set, 'bounds': bounds, 'run_minimize': True})

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)

    with Pool(max_pool) as p:
        pool_outputs = p.map(run_smac_experiment, parameters)

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('smac_results_1.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
    
    for result in pool_outputs:
        with open('smac_results_1.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(result)
