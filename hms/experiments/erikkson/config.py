from ...usc import metaepoch_limit
from ...lsc import fitness_steadiness
from ...single_pop.sea import SEA
from ...config import LevelConfig
from ...util import load_list
from ...problems.erikkson import ErikksonProblem
from ...data.erikkson_4_1_0 import data

fake_solver_config = {
    "script_path": "scripts/fake_erikkson.py",
    "solver_path": "fake_solver/build"
}

SOLVER_DIR = "/home/prac/maciej.smolka/eksperymenty/erikkson/solver/iga-ads/"

real_solver_config = {
    "script_path": SOLVER_DIR + "examples/erikkson/inverse.sh",
    "solver_path": SOLVER_DIR + "build/"
}

solver_config = real_solver_config

def erikkson(accuracy_level: int):
    return ErikksonProblem(
        script_path=solver_config["script_path"],
        solver_path=solver_config["solver_path"],
        accuracy_level=accuracy_level,
        observed_data=data
    )

bounds = [(-10, 10) for _ in range(2)]

hms_config = [
    LevelConfig(SEA(2, erikkson(0), bounds, pop_size=20)),
    LevelConfig(
        SEA(2, erikkson(1), bounds, pop_size=5, mutation_std=0.2), 
        sample_std_dev=0.1, 
        lsc=fitness_steadiness(max_deviation=0.1)
        )
]

gsc = metaepoch_limit(50)