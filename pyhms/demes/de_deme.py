from pyhms.demes.abstract_deme import AbstractDeme, DemeInitArgs
from pyhms.demes.single_pop_eas.de import DE
from pyhms.initializers import sample_normal, sample_uniform

from ..config import DELevelConfig
from ..core.individual import Individual


class DEDeme(AbstractDeme):
    def __init__(
        self,
        deme_init_args: DemeInitArgs,
    ) -> None:
        super().__init__(deme_init_args)
        config: DELevelConfig = deme_init_args.config  # type: ignore[assignment]
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._sample_std_dev = config.sample_std_dev
        self._de = DE(
            use_dither=config.dither,
            crossover_probability=config.crossover,
            f=config.scaling,
        )

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
            offspring = self._de.run(self.current_population)

            epoch_counter += 1
            metaepoch_generations.append(offspring)

            if tree._gsc(tree):
                self._history.append(metaepoch_generations)
                self._active = False
                self.log("DE Deme finished due to GSC")
                return
        self._history.append(metaepoch_generations)
        if self._lsc(self):
            self.log("DE Deme finished due to LSC")
            self._active = False
