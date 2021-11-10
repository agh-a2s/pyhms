import numpy as np

def print_pop(population):
    for ind in population:
        print(ind)

def compute_centroid(population) -> np.array:
    return np.mean([ind.genome for ind in population], axis=0)

def compute_avg_fitness(population) -> float:
    return np.mean([ind.fitness for ind in population])
