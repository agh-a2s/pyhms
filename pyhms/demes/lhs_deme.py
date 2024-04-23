from scipy.stats.qmc import LatinHypercube

from ..config import LHSLevelConfig
from ..core.individual import Individual
from ..logging_ import FilteringBoundLogger
from .abstract_deme import AbstractDeme


class LHSDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: LHSLevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual = None,
        random_seed: int = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, sprout_seed)
        self._pop_size = config.pop_size
        self.sampler = LatinHypercube(d=len(config.bounds), seed=random_seed)
        self.run()

    def run(self) -> None:
        population = [Individual(genome, problem=self._problem) for genome in self.sampler.random(self._pop_size)]
        Individual.evaluate_population(population)
        self._history.append([population])

    def run_metaepoch(self, tree) -> None:
        self.run()
        if (gsc_value := tree._gsc(tree)) or self._lsc(self):
            self._active = False
            message = "LHS Deme finished due to GSC" if gsc_value else "LHS Deme finished due to LSC"
            self.log(message)
            return
