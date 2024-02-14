from typing import List

import numpy as np
from leap_ec.individual import Individual


def compute_centroid(population: List[Individual]) -> np.ndarray:
    return np.mean([ind.genome for ind in population], axis=0)
