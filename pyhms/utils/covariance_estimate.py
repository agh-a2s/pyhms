import numpy as np

from ..core.individual import Individual
from ..demes.abstract_deme import AbstractDeme

N_INDIVIDUALS_PER_DIMENSION = 25
_EPS = 1e-10


def find_closest_rows(X: np.ndarray, y: np.ndarray, top_n: int) -> np.ndarray:
    distances = np.sqrt(((X - y) ** 2).sum(axis=1))
    closest_indices = np.argsort(distances)
    top_indices = closest_indices[:top_n]
    return X[top_indices]


def estimate_covariance(X: np.ndarray) -> np.ndarray:
    return np.cov(X.T, bias=1)  # type: ignore[call-overload]


def estimate_sigma0(X: np.ndarray) -> float:
    cov_estimate = estimate_covariance(X)
    sigma0 = np.sqrt(np.trace(cov_estimate) / len(cov_estimate))
    # Avoid ZeroDivisionError in cma.CMAEvolutionStrategy.
    return max(sigma0, _EPS)


def estimate_stds(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(estimate_covariance(X)))


def get_population(
    parent_deme: AbstractDeme,
    x0: Individual,
    use_closest_rows: bool | None = True,
) -> np.ndarray:
    n_individuals = N_INDIVIDUALS_PER_DIMENSION * len(x0.genome)
    parent_population = np.array([ind.genome for pop in parent_deme.history for ind in pop])
    if use_closest_rows:
        population = find_closest_rows(parent_population, x0.genome, n_individuals)
    else:
        population = parent_population[-n_individuals:]
    return population


def get_initial_sigma0(
    parent_deme: AbstractDeme,
    x0: Individual,
    use_closest_rows: bool | None = True,
) -> float:
    population = get_population(parent_deme, x0, use_closest_rows)
    return estimate_sigma0(population)


def get_initial_stds(
    parent_deme: AbstractDeme,
    x0: Individual,
    use_closest_rows: bool | None = True,
) -> np.ndarray:
    population = get_population(parent_deme, x0, use_closest_rows)
    return estimate_stds(population)
