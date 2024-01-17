from datetime import datetime
from typing import List, Tuple

import numpy as np
from leap_ec.individual import Individual


def print_pop(population):
    for ind in population:
        print(ind)


def compute_centroid(population: List[Individual]) -> np.array:
    return np.mean([ind.genome for ind in population], axis=0)


def compute_avg_fitness(population: List[Individual]) -> float:
    return np.mean([ind.fitness for ind in population])


def str_to_list(in_str: str) -> List[float]:
    return [float(s) for s in in_str.split()]


def load_list(file_name: str) -> List[float]:
    with open(file_name, "r") as f:
        s = f.readline()

    return str_to_list(s)


def unique_file_name(prefix, ext):
    dt_now = datetime.now()
    dt_part = dt_now.strftime("-%Y%m%d-%H%M%S")
    return prefix + dt_part + ext


def bounds_to_extent(bounds: List[Tuple[float]]) -> Tuple[float]:
    return bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
