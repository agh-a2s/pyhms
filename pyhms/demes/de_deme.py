import numpy as np
import numpy.typing as npt
from pyhms.config import DELevelConfig
from pyhms.demes.abstract_deme import AbstractDeme
from structlog.typing import FilteringBoundLogger

from ..core.individual import Individual
from ..core.initializers import PopInitializer


class DEDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: DELevelConfig,
        initializer: PopInitializer,
        logger: FilteringBoundLogger,
        started_at: int = 0,
    ) -> None:
        super().__init__(id, level, config, initializer, logger, started_at)
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._dither = config.dither
        self._scaling = config.scaling
        self._crossover_prob = config.crossover
        self._sample_std_dev = config.sample_std_dev

        starting_pop = self._initializer.sample_pop(self._pop_size, self._problem)
        Individual.evaluate_population(starting_pop)
        self._history.append([starting_pop])

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        metaepoch_generations = []
        while epoch_counter < self._generations:
            donors = self._create_donor_vectors(np.array([ind.genome for ind in self.current_population]))
            donors_pop = [Individual(donor, problem=self._problem) for donor in donors]
            Individual.evaluate_population(donors_pop)
            offspring = [self._crossover(parent, donor) for parent, donor in zip(self.current_population, donors_pop)]

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

    def _create_donor_vectors(self, parents: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        randoms = parents[np.random.randint(0, len(parents), size=(len(parents), 2))]
        if self._dither:
            scaling = np.random.uniform(0.5, 1, size=len(parents))
            scaling = np.repeat(scaling[:, np.newaxis], len(self._bounds), axis=1)
            donor = parents + scaling * (randoms[:, 0] - randoms[:, 1])
        else:
            donor = parents + self._scaling * (randoms[:, 0] - randoms[:, 1])

        # Apply mirror method for correction of boundary violations
        donor = np.where(donor < self._bounds[:, 0], 2 * self._bounds[:, 0] - donor, donor)
        donor = np.where(donor > self._bounds[:, 1], 2 * self._bounds[:, 1] - donor, donor)

        return donor

    def _crossover(self, parent: Individual, donor: Individual) -> Individual:
        if parent > donor:
            return parent
        else:
            genome = np.array(
                [p if np.random.uniform() < self._crossover_prob else d for p, d in zip(parent.genome, donor.genome)]
            )
            offspring = Individual(genome, problem=self._problem)
            offspring.evaluate()
            return offspring
