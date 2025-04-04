from ..config import SHADELevelConfig
from ..core.individual import Individual
from ..initializers import sample_normal, sample_uniform
from .abstract_deme import AbstractDeme, DemeInitArgs
from .single_pop_eas.de import SHADE


class SHADEDeme(AbstractDeme):
    def __init__(
        self,
        deme_init_args: DemeInitArgs,
    ) -> None:
        super().__init__(deme_init_args)
        config: SHADELevelConfig = deme_init_args.config  # type: ignore[assignment]
        self._init_pop_size = config.pop_size
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._sample_std_dev = config.sample_std_dev
        self._shade = SHADE(config.memory_size, config.pop_size)

        if deme_init_args.sprout_seed is None:
            starting_pop = Individual.create_population(
                self._pop_size,
                initialize=sample_uniform(bounds=self._bounds),
                problem=self._problem,
            )
        else:
            x0 = deme_init_args.sprout_seed.genome
            starting_pop = Individual.create_population(
                self._pop_size - 1,
                initialize=sample_normal(x0, self._sample_std_dev, bounds=self._bounds),
                problem=self._problem,
            )
            seed_ind = Individual(x0, problem=self._problem)
            starting_pop.append(seed_ind)

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
