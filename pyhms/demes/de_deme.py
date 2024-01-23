import numpy as np
import numpy.typing as npt
from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
from leap_ec.real_rep.initializers import create_real_vector
from pyhms.config import DELevelConfig
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.initializers import sample_normal
from structlog.typing import FilteringBoundLogger


class DEDeme(AbstractDeme):
    def __init__(
        self,
        id: str,
        level: int,
        config: DELevelConfig,
        logger: FilteringBoundLogger,
        started_at: int = 0,
        seed: Individual = None,
    ) -> None:
        super().__init__(id, level, config, logger, started_at, seed)
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._dither = config.dither
        self._scaling = config.scaling
        self._crossover_prob = config.crossover

        if seed is None:
            starting_pop = Individual.create_population(
                self._pop_size,
                initialize=create_real_vector(bounds=self._bounds),
                decoder=IdentityDecoder(),
                problem=self._problem,
            )
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
            donors = self._create_donor_vectors(np.array([ind.genome for ind in self.current_population]))
            donors_pop = [Individual(donor, problem=self._problem, decoder=IdentityDecoder()) for donor in donors]
            Individual.evaluate_population(donors_pop)
            offspring = [self._crossover(parent, donor) for parent, donor in zip(self.current_population, donors_pop)]

            epoch_counter += 1

            if tree._gsc(tree):
                self._active = False
                self._history.append(offspring)
                return

        self._history.append(offspring)
        if self._lsc(self):
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
            offspring = Individual(genome, problem=self._problem, decoder=IdentityDecoder())
            offspring.evaluate()
            return offspring
