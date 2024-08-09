from pyhms.config import EALevelConfig
from pyhms.core.individual import Individual
from pyhms.core.initializers import PopInitializer
from pyhms.demes.abstract_deme import AbstractDeme
from structlog.typing import FilteringBoundLogger


class EADeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: EALevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
    ) -> None:
        super().__init__(id, level, config, logger, started_at)
        self._sample_std_dev = config.sample_std_dev
        self._pop_size = config.pop_size
        self._generations = config.generations
        # In order to count evaluations we need to use internal problem:
        ea_params = config.__dict__.copy()
        ea_params["problem"] = self._problem
        self._ea = config.ea_class.create(**ea_params)

        starting_pop = self._initializer(self._pop_size, self._problem)
        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        metaepoch_generations = []
        while epoch_counter < self._generations:
            offspring = self._ea.run(self.current_population, mutation_std=self._get_mutation_std())
            epoch_counter += 1
            metaepoch_generations.append(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_generations)
                self._active = False
                self.log("EA Deme finished due to GSC")
                return
        self._history.append(metaepoch_generations)
        if self._lsc(self):
            self.log("EA Deme finished due to LSC")
            self._active = False

    def _get_mutation_std(self) -> float:
        config_dict = self.config.__dict__
        if mutation_std_step := config_dict.get("mutation_std_step"):
            return config_dict.get("mutation_std") + self.iterations_count_since_last_sprout * mutation_std_step
        return config_dict.get("mutation_std")

    def __str__(self) -> str:
        best_fitness = self.best_current_individual.fitness
        if self._sprout_seed is None:
            return f"Root deme {self.id} with best achieved fitness {best_fitness}"
        else:
            return f"""Deme {self.id}, metaepoch {self.started_at} and
            seed {self._sprout_seed.genome} with best {best_fitness}"""
