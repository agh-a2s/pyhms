import numpy as np
from cma import CMAEvolutionStrategy
from pyhms.core.individual import Individual
from structlog.typing import FilteringBoundLogger

from ..config import CMALevelConfig
from ..utils.covariance_estimate import get_initial_sigma0, get_initial_stds
from .abstract_deme import AbstractDeme


class CMADeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: CMALevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        random_seed: int = None,
        parent_deme: AbstractDeme | None = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at)
        self.generations = config.generations
        lb = [bound[0] for bound in config.bounds]
        ub = [bound[1] for bound in config.bounds]
        opts = {"bounds": [lb, ub], "verbose": -9}
        x0 = self._initializer.get_seed(self._problem)
        if random_seed is not None:
            opts["randn"] = np.random.randn
            opts["seed"] = random_seed + self._started_at
        if config.__dict__.get("set_stds"):
            opts["CMA_stds"] = get_initial_stds(parent_deme, x0)
            # We recommend to use sigma0 = 1 in this case.
            sigma0 = 1.0 if config.sigma0 is None else config.sigma0
            self._cma_es = CMAEvolutionStrategy(x0.genome, sigma0, inopts=opts)
        elif config.sigma0:
            self._cma_es = CMAEvolutionStrategy(x0.genome, config.sigma0, inopts=opts)
        else:
            sigma0 = get_initial_sigma0(parent_deme, x0)
            self._cma_es = CMAEvolutionStrategy(x0.genome, sigma0, inopts=opts)

        starting_pop = [Individual(solution, problem=self._problem) for solution in self._cma_es.ask()]
        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        genomes = [ind.genome for ind in self.current_population]
        values = [ind.fitness for ind in self.current_population]
        metaepoch_generations = []
        while epoch_counter < self.generations:
            self._cma_es.tell(genomes, values)
            offspring = [Individual(solution, problem=self._problem) for solution in self._cma_es.ask()]
            Individual.evaluate_population(offspring)
            genomes = [ind.genome for ind in offspring]
            values = [ind.fitness for ind in offspring]
            epoch_counter += 1
            metaepoch_generations.append(offspring)
            if (gsc_value := tree._gsc(tree)) or self._cma_es.stop():
                self._history.append(metaepoch_generations)
                self._active = False
                self._centroid = None
                message = "CMA Deme finished due to GSC" if gsc_value else "CMA Deme finished due to CMA ES stop"
                self.log(message)
                return
        self._centroid = None
        self._history.append(metaepoch_generations)

        if self._lsc(self) or self._cma_es.stop():
            self.log("CMA Deme finished due to LSC")
            self._active = False
