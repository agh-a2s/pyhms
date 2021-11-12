import logging
from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
import numpy as np

from .initializers import sample_normal
from .config import LevelConfig
from .util import compute_avg_fitness, compute_centroid

deme_logger = logging.getLogger(__name__)
class Deme:
    def __init__(self, id: str, config: LevelConfig, started_at=1, leaf=False, 
        seed=None) -> None:
        self._id = id
        self._ea = config.ea
        self._sample_std_dev = config.sample_std_dev
        self._lsc = config.lsc
        self._problem = config.ea.problem
        self._bounds = config.ea.bounds
        self._pop_size = config.ea.pop_size
        
        if seed is None:
            self._current_pop = self._ea.run()
        else:
            x = seed.genome
            pop = Individual.create_population(
                self._pop_size - 1, 
                initialize=sample_normal(x, self._sample_std_dev, bounds=self._bounds), 
                decoder=IdentityDecoder(), 
                problem=self._problem
                )
            seed_ind = Individual(x, problem=self._problem)
            pop.append(seed_ind)
            Individual.evaluate_population(pop)
            self._current_pop = pop
        
        self._started_at = started_at
        self._history = [self._current_pop]
        self._active = True
        self._children = []
        self._leaf = leaf
        self._centroid: np.array = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self._current_pop)
        return self._centroid

    def avg_fitness(self, metaepoch=-1) -> float:
        return compute_avg_fitness(self._history[metaepoch])

    @property
    def population(self):
        if self._current_pop is not None:
            return self._current_pop
        else:
            return self._history[-1]

    @property
    def best(self):
        return max(self.population)

    @property
    def metaepoch_count(self):
        return len(self._history) - 1

    @property
    def active(self) -> bool:
        return self._active

    @property
    def history(self) -> list:
        return self._history

    @property
    def is_leaf(self) -> bool:
        return self._leaf

    def run_metaepoch(self) -> None:
        self._current_pop = self._ea.run(self._current_pop)
        self._centroid = None
        self._history.append(self._current_pop)

        if self._lsc(self):
            self._active = False
            deme_logger.debug(f"{self} stopped after {self.metaepoch_count} metaepochs")

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = max(self._current_pop)
        return f"Deme {self.id} started at {self._started_at} best fitness {bsf.fitness}"
