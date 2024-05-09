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
        range_size = upper_bounds - lower_bounds
        # Normalize genomes to start from zero and find the number of "flips" needed
        normalized_genomes = genomes - lower_bounds
        flips = np.floor_divide(normalized_genomes, range_size)
        # Calculate the position within the range after even number of flips (mirroring effect)
        mod_genomes = np.mod(normalized_genomes, range_size)
        # Even flips mean the value is within the range, odd flips mean it should be mirrored
        is_odd_flip = np.mod(flips, 2) == 1
        reflected_genomes = np.where(is_odd_flip, range_size - mod_genomes, mod_genomes)
        # Return genomes to their original positions with bounds applied
        return lower_bounds + reflected_genomes
    elif method == "toroidal":
        range_size = upper_bounds - lower_bounds
        return lower_bounds + (genomes - lower_bounds) % range_size
    else:
        raise ValueError(f"Unknown method: {method}")
