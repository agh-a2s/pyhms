from pyhms.config import EALevelConfig
from pyhms.core.individual import Individual
from pyhms.core.initializers import PopInitializer
from pyhms.demes.abstract_deme import AbstractDeme
from structlog.typing import FilteringBoundLogger


class RandomDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: EALevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
    ) -> None:
        super().__init__(id, level, config, logger, started_at)
        self._pop_size = config.pop_size
        self._bounds = config.bounds
        self.run()

    def run(self) -> None:
        population = self._initializer(self._pop_size, self._problem)
        Individual.evaluate_population(population)
        self._history.append([population])

    def run_metaepoch(self, tree) -> None:
        self.run()
        if (gsc_value := tree._gsc(tree)) or self._lsc(self):
            self._active = False
            message = (
                f"Random sampler Deme of {self._initializer.__class__.__name__} class finished due to GSC"
                if gsc_value
                else f"Random sampler Deme of {self._initializer.__class__.__name__} class finished due to LSC"
            )
            self.log(message)
            return
