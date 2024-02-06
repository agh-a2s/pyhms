from typing import Dict, List, Tuple

from leap_ec.individual import Individual
from structlog.typing import FilteringBoundLogger

from .config import TreeConfig
from .demes.abstract_deme import AbstractDeme
from .demes.ea_deme import EADeme
from .demes.initialize import init_from_config, init_root
from .logging_ import DEFAULT_LOGGING_LEVEL, get_logger
from .sprout.sprout_mechanisms import SproutMechanism


class DemeTree:
    def __init__(self, config: TreeConfig) -> None:
        self.metaepoch_count: int = 0
        self.config: TreeConfig = config
        self._gsc = config.gsc
        self._sprout_mechanism: SproutMechanism = config.sprout_mechanism
        self._logger: FilteringBoundLogger = get_logger(config.options.get("log_level", DEFAULT_LOGGING_LEVEL))

        nlevels = len(config.levels)
        if nlevels < 1:
            raise ValueError("Level number must be positive")

        self._levels: List[List[AbstractDeme]] = [[] for _ in range(nlevels)]
        root_deme = init_root(config.levels[0], self._logger)
        self._levels[0].append(root_deme)

    @property
    def levels(self):
        return self._levels

    @property
    def height(self) -> int:
        return len(self.levels)

    @property
    def root(self):
        return self.levels[0][0]

    @property
    def all_demes(self) -> List[Tuple[int, AbstractDeme]]:
        return [(level_no, deme) for level_no in range(self.height) for deme in self.levels[level_no]]

    @property
    def leaves(self) -> List[AbstractDeme]:
        return self.levels[-1]

    @property
    def active_demes(self) -> List[Tuple[int, AbstractDeme]]:
        return [(level_no, deme) for level_no in range(self.height) for deme in self.levels[level_no] if deme.is_active]

    @property
    def active_non_leaves(self) -> List[Tuple[int, AbstractDeme]]:
        return [
            (level_no, deme) for level_no in range(self.height - 1) for deme in self.levels[level_no] if deme.is_active
        ]

    @property
    def best_leaf_individual(self) -> float:
        return max(deme.best_individual for deme in self.leaves)

    @property
    def best_individual(self) -> float:
        return max(deme.best_individual for level in self._levels for deme in level)

    def run(self) -> None:
        self._logger.debug(
            "Starting HMS",
            height=self.height,
            options=self.config.options,
            levels=self.config.levels,
            gsc=str(self.config.gsc),
        )
        while not self._gsc(self):
            self.metaepoch_count += 1
            self._logger = self._logger.bind(metaepoch=self.metaepoch_count)
            self.run_metaepoch()
            if not self._gsc(self):
                self.run_sprout()
            if len(self.leaves) > 0:
                self._logger.info(
                    "Metaepoch finished",
                    best_fitness=self.best_leaf_individual.fitness,
                    best_individual=self.best_leaf_individual.genome,
                )
            else:
                self._logger.info("Metaepoch finished. No leaf demes yet.")

    def run_metaepoch(self) -> None:
        for _, deme in reversed(self.active_demes):
            if "hibernation" in self.config.options and self.config.options["hibernation"] and deme._hibernating:
                continue

            deme.run_metaepoch(self)

    def run_sprout(self) -> None:
        deme_seeds = self._sprout_mechanism.get_seeds(self)
        self._do_sprout(deme_seeds)

        if "hibernation" in self.config.options and self.config.options["hibernation"]:
            for _, deme in reversed(self.active_non_leaves):
                if deme in deme_seeds:
                    if deme._hibernating:
                        self._logger.debug("Deme stopped hibernating", deme=deme.id)
                    deme._hibernating = False
                else:
                    if not deme._hibernating:
                        self._logger.debug("Deme started hibernating", deme=deme.id)
                    deme._hibernating = True

    def _do_sprout(self, deme_seeds: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]) -> None:
        for deme, info in deme_seeds.items():
            target_level = deme.level + 1

            for ind in info[1]:
                new_id = self._next_child_id(deme)
                config = self.config.levels[target_level]

                child = init_from_config(
                    config,
                    new_id,
                    target_level,
                    self.metaepoch_count,
                    seed=ind,
                    logger=self._logger,
                )
                deme.add_child(child)
                self._levels[target_level].append(child)
                self._logger.debug("Sprouted new child", seed=child._seed.genome, id=new_id, tree_level=target_level)

    def _next_child_id(self, deme: EADeme) -> str:
        if deme.level >= self.height - 1:
            raise ValueError("Only non-leaf levels are admissible")

        id_suffix = len(self._levels[deme.level + 1])
        if deme.id == "root":
            return str(id_suffix)
        else:
            return f"{deme.id}/{id_suffix}"
