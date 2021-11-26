from abc import ABC, abstractmethod
import logging
from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
import numpy as np

from .initializers import sample_normal
from .config import LevelConfig
from .util import compute_avg_fitness, compute_centroid

deme_logger = logging.getLogger(__name__)

class AbstractDeme(ABC):
    def __init__(self, id: str, started_at: int = 0) -> None:
        super().__init__()
        self._id = id
        self._started_at = started_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def started_at(self) -> int:
        return self._started_at

    @property
    @abstractmethod
    def history(self) -> list:
        raise NotImplementedError()

    @property
    def metaepoch_count(self) -> int:
        return len(self.history) - 1

    @property
    def best(self):
        return max(self.history[-1])

    def avg_fitness(self, metaepoch=-1) -> float:
        return compute_avg_fitness(self.history[metaepoch])

class Deme(AbstractDeme):
    def __init__(self, id: str, config: LevelConfig, started_at=0, leaf=False, 
        seed=None) -> None:
        super().__init__(id, started_at)
        self._sample_std_dev = config.sample_std_dev
        self._lsc = config.lsc
        self._problem = config.problem
        self._bounds = config.bounds
        self._pop_size = config.pop_size
        self._ea = config.ea_class.create(**config.__dict__)
        
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
        
        self._history = [self._current_pop]
        self._active = True
        self._children = []
        self._leaf = leaf
        self._centroid: np.array = None

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self._current_pop)
        return self._centroid

    @property
    def population(self):
        if self._current_pop is not None:
            return self._current_pop
        else:
            return self._history[-1]

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
        return f"Deme {self.id} started at {self.started_at} best fitness {bsf.fitness}"
