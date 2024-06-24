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
        x0: Individual,
        started_at: int = 0,
        random_seed: int = None,
        parent_deme: AbstractDeme | None = None,
    ) -> None:
        if x0 is None: # Can't work around it really
            Td_tile = np.array([0.1, 0.1, 0.1, 0.34, 0.55, 0.54, 0.52, 0.53, 0.41, 0.41])
            A1_grid = np.tile(np.array([151.57, 89.4135, 36.847, 17.458, 10.618, 7.9709, 6.1632, 4.7546, 4.1182, 3.8926]), (11, 1))
            A2_grid = np.tile(np.array([0.001, 12.9696, 159.9975, 48.7086, 20.6098, 12.1711, 8.0068, 4.448, 4.2918, 3.8111]), (11, 1))
            k_grid = np.tile(np.array([1.235, 1.019, 0.693, 0.528, 0.417, 0.347, 0.3, 0.259, 0.231, 0.209]), (11, 1))
            x0_genome = np.concatenate([A1_grid.flatten(), A2_grid.flatten(), k_grid.flatten(), Td_tile])
            upper_bound = [200.0] * 11 * 10 * 2 + [2.0] * 11 * 10 + [5.0] * 10
            x0 = Individual(np.array([x0_genome[i] / upper_bound[i] for i in range(len(x0_genome))]), problem=config.problem)
        super().__init__(id, level, config, logger, started_at, x0)
        self.generations = config.generations
        lb = [bound[0] for bound in config.bounds]
        ub = [bound[1] for bound in config.bounds]
        opts = {"bounds": [lb, ub], "verbose": -9}
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
        if config.__dict__.get("cma_options"):
            opts.update(config.cma_options)

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
