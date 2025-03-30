from scipy.stats.qmc import Sobol

from ..config import SobolLevelConfig
from ..core.individual import Individual
from .abstract_deme import AbstractDeme, DemeInitArgs


class SobolDeme(AbstractDeme):
    def __init__(
        self,
        deme_init_args: DemeInitArgs,
    ) -> None:
        super().__init__(deme_init_args)
        config: SobolLevelConfig = deme_init_args.config  # type: ignore[assignment]
        self._pop_size = config.pop_size
        self.sampler = Sobol(d=len(config.bounds), scramble=True, seed=deme_init_args.random_seed)
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
            message = "Sobol Deme finished due to GSC" if gsc_value else "Sobol Deme finished due to LSC"
            self.log(message)
            return
