import csv

from multiprocess import Pool
from tuning import run_smac_experiment

if __name__ == '__main__':

    max_pool = 12

    evaluation_limits = [9500]
    dimensions_list = [10]
    bounds=[(-5, 5)]
    hms_level_configurations = ["sea_2", "cma_2"]
    func_sets = range(0, 5)

    conf_for_func = {0: "cma_2", 1: "sea_2", 2: "cma_2", 3: "cma_2", 4: "sea_2"}

    parameters = []
    for i in range(5):
        for evaluation_limit in evaluation_limits:
            for dimensions in dimensions_list:
                for func_set in func_sets:
                    parameters.append({'eval': evaluation_limit, 'dim': dimensions, 'config': conf_for_func[func_set], 'test_problem': func_set, 'bounds': bounds, 'run_minimize': True})
                        

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)

    with Pool(max_pool) as p:
        pool_outputs = p.map(run_smac_experiment, parameters)

    header = ['HMS variant', 'evaluation factor', 'dimensions', 'test suite', 'config', 'fitness']
    with open('smac_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
    
    for result in pool_outputs:
        with open('smac_results.csv', 'a') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(result)
