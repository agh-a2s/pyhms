from structlog.typing import FilteringBoundLogger

from ..config import SHADELevelConfig
from ..core.individual import Individual
from .abstract_deme import AbstractDeme
from .single_pop_eas.de import SHADE


class SHADEDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: SHADELevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, sprout_seed)
        self._init_pop_size = config.pop_size
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._sample_std_dev = config.sample_std_dev
        self._shade = SHADE(config.memory_size, config.pop_size)

        starting_pop = self._initializer(self._pop_size, self._problem)
        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])


    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        metaepoch_generations = []
        while epoch_counter < self._generations:
            offspring = self._shade.run(self.current_population)

            epoch_counter += 1
            metaepoch_generations.append(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_generations)
                self._active = False
                self.log("SHADE Deme finished due to GSC")
                return
        self._history.append(metaepoch_generations)
        if self._lsc(self):
            self.log("SHADE Deme finished due to LSC")
            self._active = False
