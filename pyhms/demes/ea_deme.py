from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from pyhms.config import EALevelConfig
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.initializers import sample_normal
from structlog.typing import FilteringBoundLogger


class EADeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: EALevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        sprout_seed: Individual = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, sprout_seed)
        self._sample_std_dev = config.sample_std_dev
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._ea = config.ea_class.create(**config.__dict__)

        if sprout_seed is None:
            starting_pop = self._ea.run()
        else:
            x = sprout_seed.genome
            starting_pop = Individual.create_population(
                self._pop_size - 1,
                initialize=sample_normal(x, self._sample_std_dev, bounds=self._bounds),
                decoder=IdentityDecoder(),
                problem=self._problem,
            )
            seed_ind = Individual(x, problem=self._problem)
            starting_pop.append(seed_ind)
            Individual.evaluate_population(starting_pop)

        self._history.append(starting_pop)

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        metaepoch_offspring = []
        while epoch_counter < self._generations:
            offspring = self._ea.run(self.current_population)
            epoch_counter += 1
            metaepoch_offspring.extend(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_offspring)
                self._active = False
                self.log("EA Deme finished due to GSC")
                return
        self._history.append(metaepoch_offspring)
        if self._lsc(self):
            self.log("EA Deme finished due to LSC")
            self._active = False

    def __str__(self) -> str:
        best_fitness = self.best_current_individual.fitness
        if self._sprout_seed is None:
            return f"Root deme {self.id} with best achieved fitness {best_fitness}"
        else:
            return f"""Deme {self.id}, metaepoch {self.started_at} and
            seed {self._sprout_seed.genome} with best {best_fitness}"""
