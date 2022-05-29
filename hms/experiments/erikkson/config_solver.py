from ...problem import StatsGatheringProblem
from hms.experiments.problems import ErikksonProblem
from hms.experiments.data import data

SOLVER_DIR = "/home/prac/maciej.smolka/eksperymenty/erikkson/solver/iga-ads/"

solver_config = {
    "script_path": SOLVER_DIR + "examples/erikkson/inverse.sh",
    "solver_path": SOLVER_DIR + "build/"
}

def erikkson(accuracy_level: int):
    return StatsGatheringProblem(
        ErikksonProblem(
            script_path=solver_config["script_path"],
            solver_path=solver_config["solver_path"],
            accuracy_level=accuracy_level,
            observed_data=data
        )
    )

bounds = [(-10, 10) for _ in range(2)]
