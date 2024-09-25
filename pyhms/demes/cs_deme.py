from pyhms.config import CSLevelConfig
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.demes.single_pop_eas.cs import CuckooSearchOptimizer
from pyhms.initializers import sample_normal, sample_uniform
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual


class CSDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: CSLevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual | None = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, sprout_seed)
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._cs = CuckooSearchOptimizer(
            p=config.p,
            alpha=config.alpha,
            beta=config.beta,
        )

        if sprout_seed is None:
            starting_pop = Individual.create_population(
                self._pop_size,
                initialize=sample_uniform(bounds=self._bounds),
                problem=self._problem,
            )
        else:
            x = sprout_seed.genome
            starting_pop = Individual.create_population(
                self._pop_size - 1,
                initialize=sample_normal(x, config.sample_std_dev, bounds=self._bounds),
                problem=self._problem,
            )
            seed_ind = Individual(x, problem=self._problem)
            starting_pop.append(seed_ind)

        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        metaepoch_generations = []
        while epoch_counter < self._generations:
            offspring = self._cs.run(self.current_population)

            epoch_counter += 1
            metaepoch_generations.append(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_generations)
                self._active = False
                self.log("CS Deme finished due to GSC")
                return
        self._history.append(metaepoch_generations)
        if self._lsc(self):
            self.log("CS Deme finished due to LSC")
            self._active = False
