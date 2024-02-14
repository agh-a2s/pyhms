from datetime import datetime
from typing import List

import numpy as np
from leap_ec.individual import Individual


def compute_centroid(population: List[Individual]) -> np.ndarray:
    return np.mean([ind.genome for ind in population], axis=0)


def unique_file_name(prefix, ext):
    dt_now = datetime.now()
    dt_part = dt_now.strftime("-%Y%m%d-%H%M%S")
    return prefix + dt_part + ext
