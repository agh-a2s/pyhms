from ..demes.abstract_deme import AbstractDeme
import numpy as np


def format_array(solution: np.ndarray, float_format="{:#.2f}") -> str:
    solution_values = [float_format.format(sol) for sol in solution]
    return "(" + ", ".join(solution_values) + ")"


def print_deme(deme: AbstractDeme, best_fitness: float | None = None) -> None:
    deme_distinguisher = (
        "***" if best_fitness and deme.best_individual.fitness == best_fitness else ""
    )
    is_root = deme._sprout_seed is None
    new_deme = "(new_deme)" if deme.metaepoch_count <= 1 and not is_root else ""
    sprout = f"spr: {format_array(deme._sprout_seed.genome)};" if not is_root else ""
    best_fitness = f"f{format_array(deme.best_individual.genome)} = {deme.best_individual.fitness:.2e}"
    evaluations = f"evals: {deme.n_evaluations}"
    return print(
        f"{deme_distinguisher}{best_fitness} {sprout} {evaluations} {new_deme}"
    )


def print_tree_from_deme(
    deme: AbstractDeme, prefix: str | None = "", best_fitness: float | None = None
) -> None:
    for child in deme.children:
        if child.metaepoch_count == 0:
            # This deme did not participate in any metaepoch
            continue
        is_last = child == deme.children[-1]
        print(prefix, end="")
        if is_last:
            print("\u2514", end="")
        else:
            print("\u251C", end="")
        print("-- ", end="")
        print_deme(child, best_fitness)
        print_tree_from_deme(child, prefix=prefix + (" " if is_last else "|") + "   ")
