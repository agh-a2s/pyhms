from typing import List
import numpy as np

def print_pop(population):
    for ind in population:
        print(ind)

def compute_centroid(population) -> np.array:
    return np.mean([ind.genome for ind in population], axis=0)

def compute_avg_fitness(population) -> float:
    return np.mean([ind.fitness for ind in population])

def str_to_list(in_str: str) -> List[float]:
    return [float(s) for s in in_str.split()]

def load_list(file_name) -> List[float]:
    with open(file_name, "r") as f:
        s = f.readline()

    return str_to_list(s)