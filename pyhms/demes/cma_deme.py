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

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        individuals = []
        while epoch_counter < self.generations:
            solutions = self._cma_es.ask()
            individuals = [Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in solutions]
            self._cma_es.tell(solutions, [ind.evaluate() for ind in individuals])
            epoch_counter += 1

            if tree._gsc(tree):
                self._active = False
                self._centroid = None
                self._history.append(individuals)
                return

        self._centroid = None
        self._history.append(individuals)

        if self._lsc(self) or self._cma_es.stop():
            if self._run_minimize:
                self.run_local_optimization()
            self._active = False

    def __str__(self) -> str:
        return f"Deme {self.id}, started at metaepoch {self.started_at} with best {self.best_current_individual.fitness}"

