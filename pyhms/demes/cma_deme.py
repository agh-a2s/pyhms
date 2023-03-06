from cma import CMAEvolutionStrategy
import numpy as np
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from scipy import optimize as sopt

from .abstract_deme import AbstractDeme
from ..config import CMALevelConfig
from ..utils.misc_util import compute_centroid


class CMADeme(AbstractDeme):

    def __init__(self, id: str, config: CMALevelConfig, x0, started_at=0, leaf=False) -> None:
        super().__init__(id, started_at, config)
        self._x0 = x0
        self._sigma0 = config.sigma0
        self.generations = config.generations
        lb = [bound[0] for bound in config.bounds]
        ub = [bound[1] for bound in config.bounds]
        self._cma_es = CMAEvolutionStrategy(x0.genome, config.sigma0, inopts={'bounds': [lb, ub], 'verbose': -9})

        self._centroid = None
        self._history = [[self._x0]]
        self._active = True
        self._children = []
        self._leaf = leaf

        self._run_minimize: bool = False
        if 'run_minimize' in config.__dict__:
            self._run_minimize = config.run_minimize
        if self._run_minimize:
            argnames = set(sopt.minimize.__code__.co_varnames) - {'x0', 'fun'}
            self._minimize_args = {k: v for k, v in config.__dict__.items() if k in argnames}

    @property
    def history(self) -> list:
        return self._history

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            self._centroid = compute_centroid(self._history[-1])
        return self._centroid

    @property
    def is_leaf(self) -> bool:
        return True

    def run_metaepoch(self) -> None:
        epoch_counter = 0
        individuals = []
        while epoch_counter < self.generations:
            solutions = self._cma_es.ask()
            individuals = [Individual(solution, problem=self._problem, decoder=IdentityDecoder()) for solution in solutions]
            self._cma_es.tell(solutions, [ind.evaluate() for ind in individuals])
            epoch_counter += 1

        self._centroid = None
        self._history.append(individuals)

        if self._lsc(self) or self._cma_es.stop():
            if self._run_minimize:
                self.run_local_optimization()
            self._active = False
            # deme_logger.debug(f"{self} stopped after {self.metaepoch_count} metaepochs")

    def run_local_optimization(self) -> None:
        x0 = self.best.genome
        fun = self._problem.evaluate
        try:
            maximize = self._problem.maximize
        except AttributeError:
            maximize = False
        if maximize:
            fun = lambda x: -fun(x)

        # deme_logger.debug(f"Running minimize() from {x0} with args {self._minimize_args}")
        res = sopt.minimize(fun, x0, **self._minimize_args)
        # deme_logger.debug(f"minimize() result: {res}")

        if res.success:
            opt_ind = Individual(res.x, problem=self._problem)
            opt_ind.fitness = res.fun
            if maximize:
                opt_ind.fitness = -res.fun
            self.history[-1].append(opt_ind)

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = max(self._history[-1])
        return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._x0.genome} with best {bsf}"

