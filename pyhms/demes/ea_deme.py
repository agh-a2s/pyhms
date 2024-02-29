from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from scipy import optimize as sopt

from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.config import EALevelConfig
from pyhms.initializers import sample_normal


class EADeme(AbstractDeme):
    def __init__(self, id: str, level: int, config: EALevelConfig, started_at: int =0, seed: Individual =None) -> None:
        super().__init__(id, level, config, started_at, seed)
        self._sample_std_dev = config.sample_std_dev
        self._pop_size = config.pop_size
        self._generations = config.generations
        self._ea = config.ea_class.create(**config.__dict__)

        if seed is None:
            starting_pop = self._ea.run()
        else:
            x = seed.genome
            starting_pop = Individual.create_population(
                self._pop_size - 1,
                initialize=sample_normal(x, self._sample_std_dev, bounds=self._bounds),
                decoder=IdentityDecoder(),
                problem=self._problem
                )
            seed_ind = Individual(x, problem=self._problem)
            starting_pop.append(seed_ind)
            Individual.evaluate_population(starting_pop)

        self._history.append(starting_pop)

        self._run_minimize: bool = False
        if 'run_minimize' in config.__dict__:
            self._run_minimize = config.run_minimize
        if self._run_minimize:
            argnames = set(sopt.minimize.__code__.co_varnames) - {'x0', 'fun'}
            self._minimize_args = {k: v for k, v in config.__dict__.items() if k in argnames}

    def run_metaepoch(self, tree) -> None:
        epoch_counter = 0
        while epoch_counter < self._generations:
            offspring = self._ea.run(self.current_population)
            epoch_counter += 1

            if tree._gsc(tree):
                self._active = False
                self._history.append(offspring)
                return

        self._history.append(offspring)
        if self._lsc(self):
            self._active = False
            if self._run_minimize: self.run_local_optimization()

    def run_local_optimization(self) -> None:
        x0 = self.best_current_individual.genome
        fun = self._problem.evaluate
        try:
            res = sopt.minimize(fun, x0, **self._minimize_args)
        except RuntimeWarning:
            print("Invalid value encoutered: ", res.x, res.fun)

        opt_ind = Individual(res.x, problem=self._problem)
        opt_ind.fitness = res.fun
        self.current_population.append(opt_ind)

    def __str__(self) -> str:
        if self._seed is None:
            return f"Root deme {self.id} with best achieved fitness {self.best_current_individual.fitness}"
        else:
            return f"Deme {self.id}, metaepoch {self.started_at} and seed {self._seed.genome} with best {self.best_current_individual.fitness}"
