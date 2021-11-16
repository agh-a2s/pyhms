import logging
from hms.persist.tree import DemeTreeData

from hms.usc import metaepoch_limit

from ...algorithm import hms
from ...lsc import all_children_stopped, fitness_steadiness
from ...single_pop.sea import SEA
from ...config import LevelConfig
from ...util import load_list
from ...problems.erikkson import ErikksonProblem

DATA_FILE = "hms/data/erikkson-4-1-0.txt"

fake_solver_config = {
    "script_path": "scripts/fake_erikkson.py",
    "solver_path": "fake_solver/build"
}

real_solver_config = {
    "script_path": "../solver/iga-ads/examples/erikkson/inverse.sh",
    "solver_path": "../solver/iga-ads/build/"
}

solver_config = real_solver_config

def erikkson(accuracy_level: int):
    return ErikksonProblem(
        script_path=solver_config["script_path"],
        solver_path=solver_config["solver_path"],
        accuracy_level=1,
        observed_data=load_list(DATA_FILE)
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

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=metaepoch_limit(50))
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()