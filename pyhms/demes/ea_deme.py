import numpy as np
from scipy import optimize as sopt
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.individual import Individual

from .abstract_deme import AbstractDeme
from .deme_config import EALevelConfig
from ..operators.initializers import NormalSampling
from ..operators.callbacks import HistoryCallback
from ..utils.misc_util import compute_centroid


class EADeme(AbstractDeme):
    def __init__(self, id: str, config: EALevelConfig, started_at: int = 0, leaf: bool = False, seed=None) -> None:
        super().__init__(id, config, started_at, leaf)

        self._seed = seed
        if self._seed is None:
            self._sampler = FloatRandomSampling()
        else:
            self._sampler = NormalSampling(center=self._seed.get("X"), std_dev=config.sample_std_dev)
        
        self._ea = GA(pop_size=config.pop_size, sampling=self._sampler, eliminate_duplicates=True)
        self._ea.setup(self._problem, callback=HistoryCallback())

        self._run_minimize: bool = False
        if 'run_minimize' in config.__dict__:
            self._run_minimize = config.run_minimize
    
    @property
    def algorithm(self):
        return self._ea
    
    @property
    def history(self) -> list:
        return self._ea.callback.data["history"]
    
    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self.population)
        return self._centroid

    @property
    def population(self):
        return self._ea.pop

    def run_metaepoch(self) -> None:
        epoch_counter = 0
        while epoch_counter < self._config.generations and self._ea.has_next():
            self._ea.next()
            epoch_counter += 1

        self._centroid = None

        if self._lsc(self) or not self._ea.has_next():
            self._active = False

    def run_local_optimization(self) -> None:
        x0 = self.best.get("X")
        fun = self._problem.evaluate

        # deme_logger.debug(f"Running minimize() from {x0} with args {self._minimize_args}")
        res = sopt.minimize(fun, x0)
        # deme_logger.debug(f"minimize() result: {res}")

        if res.success:
            opt_ind = Individual(X=res.x, F=res.fun)
            self._current_pop.append(opt_ind)

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = self.best
        if self._seed is None:
            return f'Root deme {self.id} with best achieved fitness {bsf.get("F")}'
        else:
            return f'Deme {self.id}, metaepoch {self.started_at} and seed {self._seed.get("X")} with best {bsf.get("F")}'
