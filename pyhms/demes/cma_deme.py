import numpy as np
from cma import CMAEvolutionStrategy
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from structlog.typing import FilteringBoundLogger

from ..config import CMALevelConfig
from .abstract_deme import AbstractDeme


def find_closest_rows(X: np.ndarray, y: np.ndarray, top_n: int) -> np.ndarray:
    distances = np.sqrt(((X - y) ** 2).sum(axis=1))
    closest_indices = np.argsort(distances)
    top_indices = closest_indices[:top_n]
    return X[top_indices]


def estimate_covariance(X: np.ndarray) -> np.ndarray:
    return np.cov(X.T, bias=1)  # type: ignore[call-overload]


def estimate_sigma0(X: np.ndarray) -> float:
    cov_estimate = estimate_covariance(X)
    return np.sqrt(np.trace(cov_estimate) / len(cov_estimate))


def estimate_stds(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(estimate_covariance(X)))


def get_population(
    parent_deme: AbstractDeme,
    x0: Individual,
    n_individuals: int | None = 1000,
    use_closest_rows: bool | None = True,
) -> np.ndarray:
    parent_population = np.array([ind.genome for pop in parent_deme.history for ind in pop])
    if use_closest_rows:
        population = find_closest_rows(parent_population, x0.genome, n_individuals)
    else:
        population = parent_population[-n_individuals:]
    return population


def get_initial_sigma0(
    parent_deme: AbstractDeme,
    x0: Individual,
    n_individuals: int | None = 1000,
    use_closest_rows: bool | None = True,
) -> float:
    population = get_population(parent_deme, x0, n_individuals, use_closest_rows)
    return estimate_sigma0(population)


def get_initial_stds(
    parent_deme: AbstractDeme,
    x0: Individual,
    n_individuals: int | None = 1000,
    use_closest_rows: bool | None = True,
) -> np.ndarray:
    population = get_population(parent_deme, x0, n_individuals, use_closest_rows)
    return estimate_stds(population)


class CMADeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: CMALevelConfig,
        logger: FilteringBoundLogger,
        x0: Individual,
        started_at: int = 0,
        random_seed: int = None,
        parent_deme: AbstractDeme | None = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, x0)
        self.generations = config.generations
        lb = [bound[0] for bound in config.bounds]
        ub = [bound[1] for bound in config.bounds]
        opts = {"bounds": [lb, ub], "verbose": -9}
        if random_seed is not None:
            opts["randn"] = np.random.randn
            opts["seed"] = random_seed + self._started_at
        if config.sigma0:
            self._cma_es = CMAEvolutionStrategy(x0.genome, config.sigma0, inopts=opts)
        elif config.__dict__.get("set_stds"):
            sigma0 = 1.0
            opts["CMA_stds"] = get_initial_stds(parent_deme, x0)
            self._cma_es = CMAEvolutionStrategy(x0.genome, sigma0, inopts=opts)
        else:
            sigma0 = get_initial_sigma0(parent_deme, x0)
            self._cma_es = CMAEvolutionStrategy(x0.genome, sigma0, inopts=opts)

        starting_pop = [
            Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in self._cma_es.ask()
        ]
        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        genomes = [ind.genome for ind in self.current_population]
        values = [ind.fitness for ind in self.current_population]
        metaepoch_generations = []
        while epoch_counter < self.generations:
            self._cma_es.tell(genomes, values)
            offspring = [
                Individual(solution, problem=self._problem, decoder=IdentityDecoder())
                for solution in self._cma_es.ask()
            ]
            Individual.evaluate_population(offspring)
            genomes = [ind.genome for ind in offspring]
            values = [ind.fitness for ind in offspring]
            epoch_counter += 1
            metaepoch_generations.append(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_generations)
                self._active = False
                self._centroid = None
                self.log("CMA Deme finished due to GSC")
                return
        self._centroid = None
        self._history.append(metaepoch_generations)

        if self._lsc(self) or self._cma_es.stop():
            self.log("CMA Deme finished due to LSC")
            self._active = False

    @property
    def n_evaluations(self) -> int:
        return self._cma_es.result.evaluations
