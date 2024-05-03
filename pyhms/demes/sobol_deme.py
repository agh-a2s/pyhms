from scipy.stats.qmc import Sobol

from ..config import SobolLevelConfig
from ..core.individual import Individual
from ..logging_ import FilteringBoundLogger
from .abstract_deme import AbstractDeme


class SobolDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: SobolLevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual = None,
        random_seed: int = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, sprout_seed)
        self._pop_size = config.pop_size
        self.sampler = Sobol(d=len(config.bounds), scramble=True, seed=random_seed)
        self.lower_bounds = config.bounds[:, 0]
        self.upper_bounds = config.bounds[:, 1]
        self.run()

    def run(self) -> None:
        sample = self.sampler.random(self._pop_size)
        # Sobol samples also need to be scaled to the bounds
        genomes = self.lower_bounds + sample * (self.upper_bounds - self.lower_bounds)
        population = [Individual(genome, problem=self._problem) for genome in genomes]
        Individual.evaluate_population(population)
        self._history.append([population])

    def run_metaepoch(self, tree) -> None:
        self.run()
        if (gsc_value := tree._gsc(tree)) or self._lsc(self):
            self._active = False
            message = (
                "Sobol Deme finished due to GSC"
                if gsc_value
                else "Sobol Deme finished due to LSC"
            )
            self.log(message)
            return
