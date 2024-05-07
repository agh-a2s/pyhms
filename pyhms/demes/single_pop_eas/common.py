from typing import Protocol

import numpy as np

from ...core.population import Population


class VariationalOperator(Protocol):
    def __call__(self, population: Population) -> Population:
        pass


def apply_bounds(genomes: np.ndarray, bounds: np.ndarray, method: str) -> np.ndarray:
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    if method == "clip":
        return np.clip(genomes, bounds[:, 0], bounds[:, 1])
    elif method == "reflect":
        broadcasted_lower_bounds = lower_bounds + np.zeros_like(genomes)
        broadcasted_upper_bounds = upper_bounds + np.zeros_like(genomes)
        over_upper = genomes > upper_bounds
        genomes[over_upper] = 2 * broadcasted_upper_bounds[over_upper] - genomes[over_upper]
        under_lower = genomes < lower_bounds
        genomes[under_lower] = 2 * broadcasted_lower_bounds[under_lower] - genomes[under_lower]
        return genomes
    elif method == "toroidal":
        range_size = upper_bounds - lower_bounds
        return lower_bounds + (genomes - lower_bounds) % range_size
    else:
        raise ValueError(f"Unknown method: {method}")
