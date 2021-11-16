from ...problems.erikkson import ErikksonProblem
from ...data.erikkson_4_1_0 import data

solver_config = {
    "script_path": "scripts/fake_erikkson.py",
    "solver_path": "fake_solver/build"
}

def erikkson(accuracy_level: int):
    return ErikksonProblem(
        script_path=solver_config["script_path"],
        solver_path=solver_config["solver_path"],
        accuracy_level=accuracy_level,
        observed_data=data
    )

bounds = [(-10, 10) for _ in range(2)]
