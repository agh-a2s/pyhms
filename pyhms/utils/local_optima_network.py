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


class LocalOptimaNetwork:
    # TODO: Calculate weights for edges
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

    def local_minimization(self, initial_point: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            self.objective_function,
            initial_point,
            method="L-BFGS-B",
            bounds=self.bounds,
            options=self.local_method_options,
        )
        fitness_value = np.round(result.fun, self.objective_precision) if self.objective_precision else result.fun
        return result.x, fitness_value

    def perturbation(self, point: np.ndarray) -> np.ndarray:
        perturbed = point + np.random.uniform(-self.perturbation_steps, self.perturbation_steps, size=len(point))
        return np.clip(perturbed, self.bounds[:, 0], self.bounds[:, 1])

    def preprocess_solution(self, solution: np.ndarray) -> tuple:
        rounded_solution = np.round(solution, self.solution_precision) if self.solution_precision else solution
        return tuple(rounded_solution.copy())

    def add_node(self, solution: np.ndarray, value: float) -> None:
        self.graph.add_node(self.preprocess_solution(solution), value=value)

    def add_edge(self, solution1: np.ndarray, solution2: np.ndarray) -> None:
        self.graph.add_edge(self.preprocess_solution(solution1), self.preprocess_solution(solution2))

    def has_node(self, solution: np.ndarray) -> bool:
        return self.graph.has_node(self.preprocess_solution(solution))

    def run_basin_hopping(self) -> None:
        initial_point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        current_solution, current_value = self.local_minimization(initial_point)
        if not self.has_node(current_solution):
            self.add_node(current_solution, value=current_value)
        for _ in range(self.stopping_threshold):
            new_solution = self.perturbation(current_solution)
            new_solution, new_value = self.local_minimization(new_solution)

            if new_value <= current_value:
                if self.has_node(new_solution):
                    self.add_edge(current_solution, new_solution)
                else:
                    self.add_node(new_solution, new_value)
                    self.add_edge(current_solution, new_solution)
                current_solution, current_value = new_solution, new_value
            else:
                self.add_node(new_solution, new_value)

    def __call__(self) -> nx.DiGraph:
        for _ in range(self.num_of_runs):
            self.run_basin_hopping()
        return self.graph.edge_subgraph(self.graph.edges)

    def plot(self, seed: int | None = 42) -> None:
        subgraph = self.graph.edge_subgraph(self.graph.edges)
        pos = nx.spring_layout(subgraph, seed=seed)
        plt.figure(figsize=(10, 8))
        nx.draw(
            subgraph,
            pos,
            with_labels=False,
            node_size=500,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            edge_color="gray",
        )
        plt.show()
