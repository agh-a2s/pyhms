import graphviz
import numpy as np

from ..demes.abstract_deme import AbstractDeme

LEVEL_PREFIX = "-- "
LAST_LEVEL_SYMBOL = "\u2514"
MIDDLE_LEVEL_SYMBOL = "\u251c"


def format_array(solution: np.ndarray, float_format="{:#.2f}") -> str:
    solution_values = [float_format.format(sol) for sol in solution]
    return "(" + ", ".join(solution_values) + ")"


def format_deme(deme: AbstractDeme, best_fitness: float | None = None) -> str:
    best_symbol = " *** " if best_fitness and deme.best_individual.fitness == best_fitness else " "
    is_root = deme._sprout_seed is None
    root_symbol = "root" if is_root else deme._id
    new_deme_symbol = "(new_deme)" if deme.metaepoch_count <= 1 and not is_root else ""
    sprout = f" sprout: {format_array(deme._sprout_seed.genome)};" if not is_root else ""
    fitness_value = f"f{format_array(deme.best_individual.genome)} ~= {deme.best_individual.fitness:.2e}"
    evaluations = f"evals: {deme.n_evaluations}"
    deme_type = deme.__class__.__name__
    return f"{deme_type} {root_symbol}{best_symbol}{fitness_value}{sprout} {evaluations} {new_deme_symbol}"


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


def format_deme_node_label(deme: AbstractDeme, best_fitness: float | None = None) -> str:
    is_root = deme._sprout_seed is None

    deme_type = deme.__class__.__name__
    deme_id = "root" if is_root else deme._id

    label_parts = [f"{deme_type} {deme_id}"]

    fitness_value = f"Fitness: {deme.best_individual.fitness:.2e}"
    label_parts.append(fitness_value)

    genome_str = format_array(deme.best_individual.genome)
    if len(genome_str) > 30:
        genome_str = genome_str[:27] + "..."
    label_parts.append(f"Genome: {genome_str}")

    if not is_root:
        sprout_str = format_array(deme._sprout_seed.genome)
        if len(sprout_str) > 30:
            sprout_str = sprout_str[:27] + "..."
        label_parts.append(f"Sprout: {sprout_str}")

    label_parts.append(f"Evals: {deme.n_evaluations}")

    if deme.metaepoch_count <= 1 and not is_root:
        label_parts.append("(new deme)")

    return "\n".join(label_parts)


def get_node_attributes(deme: AbstractDeme, best_fitness: float | None = None) -> dict:
    is_root = deme._sprout_seed is None
    is_best = best_fitness and deme.best_individual.fitness == best_fitness
    is_new = deme.metaepoch_count <= 1 and not is_root

    attrs = {
        "shape": "box",
        "style": "filled",
        "fontname": "Arial",
    }

    deme_type = deme.__class__.__name__
    color_map = {
        "CMADeme": "#E6F3FF",
        "DEDeme": "#FFF0E6",
        "EADeme": "#E6FFE6",
        "LHSDeme": "#F3E6FF",
        "SHADEDeme": "#FFFCE6",
        "SobolDeme": "#FFE6F0",
        "LocalDeme": "#E6FFF3",
    }
    attrs["fillcolor"] = color_map.get(deme_type, "#F5F5F5")

    if is_root:
        attrs["penwidth"] = "2"
        attrs["fillcolor"] = "#FFD700"

    if is_best:
        attrs["color"] = "#FF0000"
        attrs["penwidth"] = "3"

    if is_new:
        attrs["style"] = "filled,dashed"

    return attrs


def visualize_deme_tree(root_deme: AbstractDeme, output_path: str | None = None, format: str = "pdf") -> graphviz.Digraph:
    best_fitness = None
    demes_to_check = [root_deme]
    while demes_to_check:
        deme = demes_to_check.pop()
        if deme.best_individual:
            if best_fitness is None or deme.best_individual.fitness > best_fitness:
                best_fitness = deme.best_individual.fitness
        demes_to_check.extend(deme.children)

    graph = graphviz.Digraph(
        "Deme Hierarchy",
        format=format,
        node_attr={"fontsize": "10"},
        graph_attr={"rankdir": "TB", "ranksep": "1.0"},
    )

    def add_deme_to_graph(deme, parent_id=None):
        deme_id = str(id(deme))

        if deme.metaepoch_count == 0 and parent_id is not None:
            return

        node_label = format_deme_node_label(deme, best_fitness)
        node_attrs = get_node_attributes(deme, best_fitness)
        graph.node(deme_id, node_label, **node_attrs)

        if parent_id is not None:
            graph.edge(parent_id, deme_id)

        for child in deme.children:
            add_deme_to_graph(child, deme_id)

    add_deme_to_graph(root_deme)

    if output_path:
        graph.render(output_path, cleanup=True)

    return graph
