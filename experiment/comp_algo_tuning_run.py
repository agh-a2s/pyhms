import csv

from multiprocess import Pool
from comp_algo_tuning import run_smac_experiment

if __name__ == '__main__':

    max_pool = 12

    evaluation_limits = [9500]
    dimensions_list = [10]
    bounds=[(-5, 5)]
    algo_configurations = ["ilshade", "cmaes"]
    func_sets = range(0, 5)

    parameters = []
    for i in range(5):
        for algo_con in algo_configurations:
            for evaluation_limit in evaluation_limits:
                for dimensions in dimensions_list:
                    for func_set in func_sets:
                        parameters.append({'eval': evaluation_limit, 'dim': dimensions, 'config': algo_con, 'test_problem': func_set, 'bounds': bounds, 'run_minimize': True})


    header = ['Algo', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)

    with Pool(max_pool) as p:
        pool_outputs = p.map(run_smac_experiment, parameters)

    header = ['Algo', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('smac_comp_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
    
    for result in pool_outputs:
        with open('smac_comp_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(result)
