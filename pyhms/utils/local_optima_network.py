from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

DEFAULT_LOCAL_METHOD_OPTIONS = {
    "ftol": 1e-07,
    "gtol": 1e-05,
}

DEFAULT_PERMUTATION_STEP = 0.5


def plot_network(graph: nx.DiGraph, seed: int | None = 42) -> None:
    pos = nx.spring_layout(graph, seed=seed)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        edge_color="gray",
    )
    plt.show()


class LocalOptimaNetwork:
    def __init__(
        self,
        objective_function: Callable,
        bounds: np.ndarray,
        stopping_threshold: int,
        num_of_runs: int,
        perturbation_steps: float | np.ndarray | None,
        local_method_options: dict = DEFAULT_LOCAL_METHOD_OPTIONS,
        solution_precision: int | None = None,
        objective_precision: int | None = None,
    ):
        self.objective_function = objective_function
        self.bounds = bounds
        if perturbation_steps is None:
            self.perturbation_steps = np.full(len(bounds), DEFAULT_PERMUTATION_STEP)
        elif isinstance(perturbation_steps, float):
            self.perturbation_steps = np.full(len(bounds), perturbation_steps)
        elif isinstance(perturbation_steps, np.ndarray):
            assert len(perturbation_steps) == len(bounds)
            self.perturbation_steps = perturbation_steps
        self.stopping_threshold = stopping_threshold
        self.local_method_options = local_method_options
        self.num_of_runs = num_of_runs
        self.solution_precision = solution_precision
        self.objective_precision = objective_precision
        self.graph = nx.DiGraph()

    def local_minimization(
        self, initial_point: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        result = minimize(
            self.objective_function,
            initial_point,
            method="L-BFGS-B",
            bounds=self.bounds,
            options=self.local_method_options,
        )
        fitness_value = (
            np.round(result.fun, self.objective_precision)
            if self.objective_precision
            else result.fun
        )
        return result.x, result.fun, fitness_value

    def perturbation(self, point: np.ndarray) -> np.ndarray:
        perturbed = point + np.random.uniform(
            -self.perturbation_steps, self.perturbation_steps, size=len(point)
        )
        return np.clip(perturbed, self.bounds[:, 0], self.bounds[:, 1])

    def preprocess_solution(self, solution: np.ndarray) -> tuple:
        rounded_solution = (
            np.round(solution, self.solution_precision)
            if self.solution_precision
            else solution
        )
        return tuple(rounded_solution.copy())

    def add_node(
        self, solution: np.ndarray, value: float, rounded_value: float
    ) -> None:
        self.graph.add_node(
            self.preprocess_solution(solution), value=value, rounded_value=rounded_value
        )

    def has_node(self, solution: np.ndarray) -> bool:
        return self.graph.has_node(self.preprocess_solution(solution))

    def add_or_update_edge(self, solution1: np.ndarray, solution2: np.ndarray) -> None:
        preprocessed_solution1 = self.preprocess_solution(solution1)
        preprocessed_solution2 = self.preprocess_solution(solution2)
        if self.graph.has_edge(preprocessed_solution1, preprocessed_solution2):
            self.graph[preprocessed_solution1][preprocessed_solution2]["weight"] += 1
        else:
            self.graph.add_edge(
                preprocessed_solution1, preprocessed_solution2, weight=1.0
            )

    def run_basin_hopping(self) -> None:
        initial_point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        current_solution, current_value, rounded_value = self.local_minimization(
            initial_point
        )
        if not self.has_node(current_solution):
            self.add_node(
                current_solution, value=current_value, rounded_value=rounded_value
            )
        iterations_since_last_update = 0
        while iterations_since_last_update < self.stopping_threshold:
            new_solution = self.perturbation(current_solution)
            new_solution, new_value, new_rounded_value = self.local_minimization(
                new_solution
            )

            if new_value <= current_value:
                if self.has_node(new_solution):
                    self.add_or_update_edge(current_solution, new_solution)
                else:
                    self.add_node(
                        new_solution, value=new_value, rounded_value=new_rounded_value
                    )
                    self.add_or_update_edge(current_solution, new_solution)
                current_solution, current_value = new_solution, new_value
                iterations_since_last_update = 0
            else:
                self.add_node(
                    new_solution, value=new_value, rounded_value=new_rounded_value
                )
                iterations_since_last_update += 1

    def remove_unconnected_nodes(self) -> nx.DiGraph:
        return self.graph.edge_subgraph(self.graph.edges)

    def compress_graph(self) -> nx.DiGraph:
        compressed_graph = nx.DiGraph()
        fitness_to_nodes = {}

        # Group nodes by fitness value
        for node, data in self.graph.nodes(data=True):
            fitness_value = data["rounded_value"]
            if fitness_value not in fitness_to_nodes:
                fitness_to_nodes[fitness_value] = []
            fitness_to_nodes[fitness_value].append(node)

        # Create compressed nodes
        for fitness_value, nodes in fitness_to_nodes.items():
            subgraph = self.graph.subgraph(nodes)
            components = nx.connected_components(subgraph.to_undirected())
            for component in components:
                component_nodes = list(component)
                if component_nodes:
                    compressed_node = tuple(sorted(component_nodes))
                    compressed_graph.add_node(compressed_node, value=fitness_value)

        # Add edges to the compressed graph
        for solution1, solution2, data in self.graph.edges(data=True):
            solution1_fitness = self.graph.nodes[solution1]["rounded_value"]
            solution2_fitness = self.graph.nodes[solution2]["rounded_value"]
            if solution1_fitness == solution2_fitness:
                continue
            solution1_compressed = tuple(
                sorted([node for node in compressed_graph if solution1 in node][0])
            )
            solution2_compressed = tuple(
                sorted([node for node in compressed_graph if solution2 in node][0])
            )
            if compressed_graph.has_edge(solution1_compressed, solution2_compressed):
                compressed_graph[solution1_compressed][solution2_compressed][
                    "weight"
                ] += data["weight"]
            else:
                compressed_graph.add_edge(
                    solution1_compressed, solution2_compressed, weight=data["weight"]
                )

        return compressed_graph

    def __call__(self) -> nx.DiGraph:
        for _ in range(self.num_of_runs):
            self.run_basin_hopping()
        self.graph = self.remove_unconnected_nodes()
        self.compressed_graph = self.compress_graph()
        return self.compressed_graph
