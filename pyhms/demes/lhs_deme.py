from scipy.stats.qmc import LatinHypercube

from ..config import LHSLevelConfig
from ..core.individual import Individual
from .abstract_deme import AbstractDeme, DemeInitArgs


class LHSDeme(AbstractDeme):
    def __init__(
        self,
        deme_init_args: DemeInitArgs,
    ) -> None:
        super().__init__(deme_init_args)
        config: LHSLevelConfig = deme_init_args.config  # type: ignore[assignment]
        self._pop_size = config.pop_size
        self.sampler = LatinHypercube(d=len(config.bounds), seed=deme_init_args.random_seed)
        self.lower_bounds = config.bounds[:, 0]
        self.upper_bounds = config.bounds[:, 1]
        self.run()

    def run(self) -> None:
        sample = self.sampler.random(self._pop_size)
        # LHS samples need to be scaled to the bounds
        genomes = self.lower_bounds + sample * (self.upper_bounds - self.lower_bounds)
        population = [Individual(genome, problem=self._problem) for genome in genomes]
        Individual.evaluate_population(population)
        self._history.append([population])

    def run_metaepoch(self, tree) -> None:
        self.run()
        if (gsc_value := tree._gsc(tree)) or self._lsc(self):
            self._active = False
            message = "LHS Deme finished due to GSC" if gsc_value else "LHS Deme finished due to LSC"
            self.log(message)
            return
