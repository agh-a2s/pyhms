import copy
import csv

from cma.bbobbenchmarks import instantiate

from pyhms.pyhms.hms import hms
from pyhms.pyhms.config import EALevelConfig, CMALevelConfig
from pyhms.pyhms.demes.single_pop_eas.sea import SEA
from pyhms.pyhms.problem import EvalCutoffProblem, FunctionProblem
from pyhms.pyhms.sprout import composite_condition, far_enough, deme_per_level_limit
from pyhms.pyhms.stop_conditions.gsc import singular_problem_eval_limit_reached
from pyhms.pyhms.stop_conditions.usc import metaepoch_limit, dont_stop

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


def set1_separable(dim):
    sphere = instantiate(1)[0]
    ellipsoidal = instantiate(2)[0]
    rastrigin = instantiate(3)[0]
    other_rastrigin = instantiate(4)[0]
    linear_slope = instantiate(5)[0]
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: sphere(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: ellipsoidal(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: rastrigin(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: other_rastrigin(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: linear_slope(x), maximize=False), dim*950)]

def set2_low_conditioning(dim):
    attractive_sector = instantiate(6)[0]
    step_ellipsoidal = instantiate(7)[0]
    rosenbrock = instantiate(8)[0]
    rosenbrock_rotated = instantiate(9)[0]
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: attractive_sector(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: step_ellipsoidal(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: rosenbrock(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: rosenbrock_rotated(x), maximize=False), dim*950)]

def set3_high_conditioning(dim):
    other_ellipsoidal = instantiate(10)[0]
    discus = instantiate(11)[0]
    bent_cigar = instantiate(12)[0]
    sharp_ridge = instantiate(13)[0]
    diff_powers = instantiate(14)[0]
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: other_ellipsoidal(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: discus(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: bent_cigar(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: sharp_ridge(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: diff_powers(x), maximize=False), dim*950)]

def set4_mmod_strong_struct(dim):
    rastrigin = instantiate(15)[0]
    weierstrass = instantiate(16)[0]
    schaffers_f7 = instantiate(17)[0]
    schaffers_f7_ill_cond = instantiate(18)[0]
    griewank_rosenbrock = instantiate(19)[0]
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: rastrigin(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: weierstrass(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: schaffers_f7(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: schaffers_f7_ill_cond(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: griewank_rosenbrock(x), maximize=False), dim*950)]

def set5_mmod_weak_struct(dim):
    schwefel = instantiate(20)[0]
    gallaghers_101_peaks = instantiate(21)[0]
    gallaghers_21_peaks = instantiate(22)[0]
    katsuura = instantiate(23)[0]
    lunacek_bi_rastrigin = instantiate(24)[0]
    
    return [EvalCutoffProblem(FunctionProblem(lambda x: schwefel(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: gallaghers_101_peaks(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: gallaghers_21_peaks(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: katsuura(x), maximize=False), dim*950),
            EvalCutoffProblem(FunctionProblem(lambda x: lunacek_bi_rastrigin(x), maximize=False), dim*950)]


def prepare_evaluator_sea_2(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_sea_2(config):
        gsc = singular_problem_eval_limit_reached(limit=evaluation_factor)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
        options = {'hibernation': False}

        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem)
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
            tree = hms(level_config=config_sea2, gsc=gsc, sprout_cond=sprout_cond, options=options)
            if len(tree.optima) > 0:
                minimas.append(tree.historic_best.fitness)
            else:
                return 2147483647.0
        return sum(minimas)
    return evaluator_sea_2

def prepare_evaluator_cma_2(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_cma_2(config):
        gsc = singular_problem_eval_limit_reached(limit=evaluation_factor)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
        options = {'hibernation': False}

        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem)
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
            tree = hms(level_config=config_cma2, gsc=gsc, sprout_cond=sprout_cond, options=options)
            if len(tree.optima) > 0:
                minimas.append(tree.historic_best.fitness)
            else:
                return 2147483647.0
        return sum(minimas)
    return evaluator_cma_2

def prepare_evaluator_cma_3(function_set, dim_number, bounds, evaluation_factor):
    def evaluator_cma_3(config):
        gsc = singular_problem_eval_limit_reached(limit=evaluation_factor)
        sprout_cond = composite_condition([far_enough(config["far_enough"]), deme_per_level_limit(config["level_limit"])])
        options = {'hibernation': False}

        minimas = []
        for test_problem in function_set:
            problem=copy.deepcopy(test_problem)
            config_cma3 = [
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
            tree = hms(level_config=config_cma3, gsc=gsc, sprout_cond=sprout_cond, options=options)
            if len(tree.optima) > 0:
                minimas.append(tree.historic_best.fitness)
            else:
                return 2147483647.0
        return sum(minimas)
    return evaluator_cma_3
    
def prepare_sea_2(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.2, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop2", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations1", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations2", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 5, 40))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation2", 0.01, 1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sample_dev2", 0.1, 3.0))
    
    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 2,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 2 times with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_sea_2(test_problem, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_cma_2(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.2, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations1", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations2", 3, 30))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 30, 300))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma2", 0.1, 3.0))

    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 2,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 2 times with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_cma_2(test_problem, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_cma_3(test_problem, dim_number, bounds, evaluation_factor):
    # Define hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("far_enough", 0.2, 20.0))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("level_limit", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop1", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("pop2", 20, 300))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations1", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations2", 2, 10))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("generations3", 3, 30))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch2", 5, 20))
    configspace.add_hyperparameter(UniformIntegerHyperparameter("meataepoch3", 30, 300))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation1", 0.25, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("mutation2", 0.1, 1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sample_dev2", 0.1, 3.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter("sigma3", 0.1, 2.0))

    # Meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality
        "runcount-limit": 1000,  # Max number of function evaluations
        "deterministic": "false",
        "maxR": 2,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 2 times with different seeds
        "output_dir": "smac_outputs",
        "cs": configspace,
    })
    
    evaluator = prepare_evaluator_cma_3(test_problem, dim_number, bounds, evaluation_factor)

    return SMAC4HPO(scenario=scenario, tae_runner=evaluator)

def prepare_config(hms_variant, test_problem, dim_number, bounds, evaluation_factor):
    if hms_variant == "sea_2":
        return prepare_sea_2(test_problem, dim_number, bounds, evaluation_factor)
    elif hms_variant == "cma_2":
        return prepare_cma_2(test_problem, dim_number, bounds, evaluation_factor)
    elif hms_variant == "cma_3":
        return prepare_cma_3(test_problem, dim_number, bounds, evaluation_factor)

def run_smac_experiment(parameters):
    func_sets = [set1_separable, set2_low_conditioning, set3_high_conditioning, set4_mmod_strong_struct, set5_mmod_weak_struct]
    test_problem = func_sets[parameters['test_problem']]
    experiment = prepare_config(parameters['config'], test_problem(parameters['dim']), parameters['dim'], parameters['bounds'], parameters['eval'])
    best_found_config = experiment.optimize()
    result = [parameters['config'], parameters['eval'], parameters['dim'], parameters['test_problem'], best_found_config.get_dictionary()]
    with open('backup_results.csv', 'a') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(result)
    return result
