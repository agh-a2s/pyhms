import numpy as np

DEFAULT_GENERATIONS_BY_LEVEL = {0: 1, 1: 20}
DEFAULT_GENERATIONS = 20
INITIAL_POPULATION_SIZE = 10
POPULATION_CONSTANT_PER_DIMENSION = 2


def get_default_mutation_std(bounds: np.ndarray, tree_level: int) -> float:
    range = np.mean(bounds[:, 1] - bounds[:, 0])
    mutation_std = range / 4
    return mutation_std / (tree_level + 1)


def get_default_population_size(bounds: np.ndarray, tree_level: int) -> int:
    n_dimensions = len(bounds)
    dimensionality_adjusted_population_size = INITIAL_POPULATION_SIZE + POPULATION_CONSTANT_PER_DIMENSION * n_dimensions
    return int(dimensionality_adjusted_population_size / (tree_level + 1))


def get_default_generations(bounds: np.ndarray, tree_level: int) -> int:
    return DEFAULT_GENERATIONS_BY_LEVEL.get(tree_level, DEFAULT_GENERATIONS)
