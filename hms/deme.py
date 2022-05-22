from abc import ABC, abstractmethod
import logging
from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
import numpy as np
import scipy.optimize as sopt

from .initializers import sample_normal
from .config import EALevelConfig
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
    @abstractmethod
    def centroid(self) -> np.array:
        raise NotImplementedError()

    @property
    def all_individuals(self) -> list:
        inds = []
        for pop in self.history:
            inds += pop

        return inds

    @property
    def metaepoch_count(self) -> int:
        return len(self.history) - 1

    @property
    def best(self):
        return max(self.history[-1])

    def avg_fitness(self, metaepoch=-1) -> float:
        return compute_avg_fitness(self.history[metaepoch])

class EA_Deme(AbstractDeme):
    def __init__(self, id: str, config: EALevelConfig, started_at=0, leaf=False,
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

        self._run_minimize: bool = False
        if 'run_minimize' in config.__dict__:
            self._run_minimize = config.run_minimize
        if self._run_minimize:
            argnames = set(sopt.minimize.__code__.co_varnames) - {'x0', 'fun'}
            self._minimize_args = {k: v for k, v in config.__dict__.items() if k in argnames}

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
    def best(self):
        if self._current_pop is not None:
            return max(self._current_pop)
        else:
            return super().best

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
            if self._run_minimize:
                self.run_local_optimization()

            self._active = False
            deme_logger.debug(f"{self} stopped after {self.metaepoch_count} metaepochs")

    def run_local_optimization(self) -> None:
        x0 = self.best.genome
        fun = self._problem.evaluate
        try:
            maximize = self._problem.maximize
        except AttributeError:
            maximize = False
        if maximize:
            fun = lambda x: -fun(x)

        deme_logger.debug(f"Running minimize() from {x0} with args {self._minimize_args}")
        res = sopt.minimize(fun, x0, **self._minimize_args)
        deme_logger.debug(f"minimize() result: {res}")

        if res.success:
            opt_ind = Individual(res.x, problem=self._problem)
            opt_ind.fitness = res.fun
            if maximize:
                opt_ind.fitness = -res.fun
            self._current_pop.append(opt_ind)

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = max(self._current_pop)
        return f"Deme {self.id} started at {self.started_at} with best achieved fitness {bsf.fitness}"
