from typing import Generator, List, Tuple
import logging

from .config import LevelConfig
from .deme import Deme

tree_logger = logging.getLogger(__name__)

class DemeTree:
    def __init__(self, level_config: List[LevelConfig], gsc, 
        sprout_cond=lambda deme, level, tree: True) -> None:
        if len(level_config) < 1:
            raise ValueError("Level number must be positive")

        self._level_config = level_config
        self._levels: List[List[Deme]] = [[] for _ in range(len(level_config))]
        root_deme = Deme("root", level_config[0], leaf=(self.height == 1))
        self._levels[0].append(root_deme)

        self._metaepoch_counter = 0
        self._gsc = gsc
        self._can_sprout = sprout_cond

    @property
    def height(self):
        return len(self._levels)

    @property
    def root(self):
        return self._levels[0][0]

    def level(self, no: int) -> List[Deme]:
        return self._levels[no]

    @property
    def all_demes(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height):
            for deme in self._levels[level_no]:
                yield level_no, deme

    @property
    def active_demes(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height):
            for deme in self._levels[level_no]:
                if deme.active:
                    yield level_no, deme

    @property
    def non_leaves(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height - 1):
            for deme in self._levels[level_no]:
                yield level_no, deme

    @property
    def active_non_leaves(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height - 1):
            for deme in self._levels[level_no]:
                if deme.active:
                    yield level_no, deme

    def demes(self, level_numbers) -> Generator[Tuple[int, Deme], None, None]:
        for level in level_numbers:
            for deme in self._levels[level]:
                yield level, deme

    @property
    def metaepoch_count(self) -> int:
        return self._metaepoch_counter

    def run(self):
        while not self._gsc(self):
            self._metaepoch_counter += 1
            tree_logger.debug(f"Metaepoch {self.metaepoch_count}")
            self.run_metaepoch()
            if not self._gsc(self):
                self.run_sprout()

    def run_metaepoch(self):
        for level, deme in self.active_demes:
            tree_logger.debug(f"Running metaepoch in deme {deme} at level {level}")
            deme.run_metaepoch()

    def run_sprout(self):
        for level, deme in self.active_non_leaves:
            if self._can_sprout(deme, level, self):
                self._do_sprout(deme, level)

    def _do_sprout(self, deme, level):
        new_id = self._next_child_id(deme, level)
        is_leaf = (level == self.height - 1)
        child = Deme(
            id=new_id, 
            config=self._level_config[level + 1], 
            started_at=self.metaepoch_count, 
            leaf=is_leaf,
            seed=max(deme.population)
            )
        deme.add_child(deme)
        self._levels[level + 1].append(child)


    def _next_child_id(self, deme: Deme, level: int) -> str:
        if level >= self.height - 1:
            raise ValueError("Only non-leaf levels are admissible")

        id_suffix = len(self._levels[level + 1])
        if deme.id == "root":
            return str(id_suffix)
        else:
            return f"{deme.id}/{id_suffix}"
