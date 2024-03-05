import numpy as np

from ..demes.abstract_deme import AbstractDeme

LEVEL_PREFIX = "-- "
LAST_LEVEL_SYMBOL = "\u2514"
MIDDLE_LEVEL_SYMBOL = "\u251C"


def format_array(solution: np.ndarray, float_format="{:#.2f}") -> str:
    solution_values = [float_format.format(sol) for sol in solution]
    return "(" + ", ".join(solution_values) + ")"


def format_deme(deme: AbstractDeme, best_fitness: float | None = None) -> str:
    best_symbol = "*** " if best_fitness and deme.best_individual.fitness == best_fitness else ""
    is_root = deme._sprout_seed is None
    root_symbol = "root" if is_root else deme._id
    new_deme_symbol = "(new_deme)" if deme.metaepoch_count <= 1 and not is_root else ""
    sprout = f" sprout: {format_array(deme._sprout_seed.genome)};" if not is_root else ""
    fitness_value = f"f{format_array(deme.best_individual.genome)} ~= {deme.best_individual.fitness:.2e}"
    evaluations = f"evals: {deme.n_evaluations}"
    deme_type = deme.__class__.__name__
    return f"{deme_type} {root_symbol} {best_symbol} {fitness_value}{sprout} {evaluations} {new_deme_symbol}"


def format_deme_children_tree(deme: AbstractDeme, prefix: str | None = "", best_fitness: float | None = None) -> str:
    formatted_tree = ""
    for child in deme.children:
        if child.metaepoch_count == 0:
            # This deme did not participate in any metaepoch
            continue
        is_last = child == deme.children[-1]
        deme_prefix = prefix + (LAST_LEVEL_SYMBOL if is_last else MIDDLE_LEVEL_SYMBOL) + LEVEL_PREFIX
        formatted_deme = format_deme(child, best_fitness)
        formatted_children_tree = format_deme_children_tree(
            child,
            prefix=prefix + (" " if is_last else "|") + "   ",
            best_fitness=best_fitness,
        )
        formatted_tree = f"{formatted_tree}{deme_prefix}{formatted_deme}\n{formatted_children_tree}"
    return formatted_tree
