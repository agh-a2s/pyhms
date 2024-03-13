import numpy as np
from cma import CMAEvolutionStrategy
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from sklearn.covariance import empirical_covariance
from structlog.typing import FilteringBoundLogger

from ..config import CMALevelConfig
from .abstract_deme import AbstractDeme


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

        if config.sigma0 is None:
            last_5_generation_population = np.array([ind.genome for pop in parent_deme.history[-5:] for ind in pop])
            cov_estimate = empirical_covariance(last_5_generation_population)
            sigma0 = np.sqrt(np.trace(cov_estimate) / len(cov_estimate))
        else:
            sigma0 = config.sigma0
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
