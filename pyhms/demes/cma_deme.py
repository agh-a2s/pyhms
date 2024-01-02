from cma import CMAEvolutionStrategy
import numpy as np
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder

from .abstract_deme import AbstractDeme
from ..config import CMALevelConfig
from ..utils.misc_util import compute_centroid


class CMADeme(AbstractDeme):

    def __init__(self, id: str, level: int, config: CMALevelConfig, x0: Individual, started_at: int=0) -> None:
        super().__init__(id, level, config, started_at)
        self.generations = config.generations
        lb = [bound[0] for bound in config.bounds]
        ub = [bound[1] for bound in config.bounds]
        self._cma_es = CMAEvolutionStrategy(x0.genome, config.sigma0, inopts={'bounds': [lb, ub], 'verbose': -9})

        self._active = True
        self._children = []
        starting_pop = [Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in self._cma_es.ask()]
        Individual.evaluate_population(starting_pop)
        self._history.append(starting_pop)

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        genomes = [ind.genome for ind in self.current_population]
        values = [ind.fitness for ind in self.current_population]
        while epoch_counter < self.generations:
            self._cma_es.tell(genomes, values)
            offspring = [Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in self._cma_es.ask()]
            Individual.evaluate_population(offspring)
            genomes = [ind.genome for ind in offspring]
            values = [ind.fitness for ind in offspring]
            epoch_counter += 1

            if tree._gsc(tree):
                self._active = False
                self._centroid = None
                self._history.append(offspring)
                return
        self._centroid = None
        self._history.append(offspring)

        if self._lsc(self) or self._cma_es.stop():
            self._active = False

    def __str__(self) -> str:
        return f"Deme {self.id}, started at metaepoch {self.started_at} with best {self.best_current_individual.fitness}"

