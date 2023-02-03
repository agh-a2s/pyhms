import numpy as np
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from scipy import optimize as sopt

from .abstract_deme import AbstractDeme
from ..config import EALevelConfig
from ..initializers import sample_normal
from ..utils.misc_util import compute_centroid


class EADeme(AbstractDeme):
    def __init__(self, id: str, config: EALevelConfig, started_at=0, leaf=False,
                 seed=None) -> None:
        super().__init__(id, started_at, config)
        self._sample_std_dev = config.sample_std_dev
        self._pop_size = config.pop_size
        self._ea = config.ea_class.create(**config.__dict__)
        self._seed = seed

        if seed is None:
            self._current_pop = self._ea.run()
        elif seed == "inject":
            pass
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
        self._children = []
        self._leaf = leaf
        self._centroid: np.array = None

        # TODO: Create local optimization deme
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
            self._current_pop.append(opt_ind)

    def add_child(self, deme):
        self._children.append(deme)

    @property
    def children(self):
        return self._children

    def __str__(self) -> str:
        bsf = max(self._current_pop)
        if self._seed is None:
            return f"Root deme {self.id} with best achieved fitness {bsf}"
        else:
            return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._seed.genome} with best {bsf}"
