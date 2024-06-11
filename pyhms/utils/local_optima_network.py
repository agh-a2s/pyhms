from typing import Callable

import networkx as nx
import numpy as np
from scipy.optimize import minimize

DEFAULT_LOCAL_METHOD_OPTIONS = {
    "ftol": 1e-07,
    "gtol": 1e-05,
}

DEFAULT_PERMUTATION_STEP = 0.5


class LocalOptimaNetwork:
    def __init__(
        self,
        objective_function: Callable,
        bounds: np.ndarray,
        stopping_threshold: int,
        num_of_runs: int,
        perturbation_steps: float | np.ndarray | None,
        local_method_options: dict = DEFAULT_LOCAL_METHOD_OPTIONS,
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
        self.graph = nx.DiGraph()

    def local_minimization(self, initial_point: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            self.objective_function,
            initial_point,
            method="L-BFGS-B",
            bounds=self.bounds,
            options=self.local_method_options,
        )
        return result.x, result.fun

    def perturbation(self, point: np.ndarray) -> np.ndarray:
        perturbed = point + np.random.uniform(-self.perturbation_steps, self.perturbation_steps, size=len(point))
        return np.clip(perturbed, self.bounds[:, 0], self.bounds[:, 1])

    def __call__(self) -> nx.DiGraph:
        # TODO: initialize solution for every run separately

        initial_point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        current_solution, current_value = self.local_minimization(initial_point)

        self.graph.add_node(tuple(current_solution.copy()), value=current_value)

        for _ in range(self.num_of_runs):
            r = 0
            while r < self.stopping_threshold:
                new_solution = self.perturbation(current_solution)
                new_solution, new_value = self.local_minimization(new_solution)

                if new_value <= current_value:
                    if self.graph.has_node(tuple(new_solution.copy())):
                        self.graph.add_edge(tuple(current_solution.copy()), tuple(new_solution.copy()))
                    else:
                        self.graph.add_node(tuple(new_solution.copy()), value=new_value)
                        self.graph.add_edge(tuple(current_solution.copy()), tuple(new_solution.copy()))
                    current_solution, current_value = new_solution, new_value
                else:
                    self.graph.add_node(tuple(new_solution.copy()), value=new_value)
                r += 1

        return self.graph.edge_subgraph(self.graph.edges)
