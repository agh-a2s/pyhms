from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from pyhms.config import EALevelConfig
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.initializers import sample_normal


class EADeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: EALevelConfig,
        started_at: int = 0,
        seed: Individual = None,
    ) -> None:
        super().__init__(id, level, config, started_at, seed)
        self._sample_std_dev = config.sample_std_dev
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._ea = config.ea_class.create(**config.__dict__)

        if seed is None:
            starting_pop = self._ea.run()
        else:
            x = seed.genome
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
        while epoch_counter < self._generations:
            offspring = self._ea.run(self.current_population)
            epoch_counter += 1

            if tree._gsc(tree):
                self._active = False
                self._history.append(offspring)
                return

        self._history.append(offspring)
        if self._lsc(self):
            self._active = False

    def __str__(self) -> str:
        best_fitness = self.best_current_individual.fitness
        if self._seed is None:
            return f"Root deme {self.id} with best achieved fitness {best_fitness}"
        else:
            return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._seed.genome} with best {best_fitness}"
