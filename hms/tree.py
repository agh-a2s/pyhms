from abc import ABC, abstractmethod
from typing import Generator, List, Tuple
import logging

from .config import TreeConfig
from .deme import AbstractDeme, Deme

tree_logger = logging.getLogger(__name__)

class AbstractDemeTree(ABC):
    def __init__(self, metaepoch_count: int) -> None:
        super().__init__()
        self._metaepoch_count = metaepoch_count

    @property
    def metaepoch_count(self) -> int:
        return self._metaepoch_count

    @property
    @abstractmethod
    def levels(self) -> List[List[AbstractDeme]]:
        raise NotImplementedError()

    def level(self, no: int) -> List[AbstractDeme]:
        return self.levels[no]

    @property
    def height(self) -> int:
        return len(self.levels)

    @property
    def root(self):
        return self.levels[0][0]

    @property
    def leaves(self) -> List[AbstractDeme]:
        return self.levels[-1]

    @property
    def non_leaves(self) -> Generator[Tuple[int, AbstractDeme], None, None]:
        for level_no in range(self.height - 1):
            for deme in self.levels[level_no]:
                yield level_no, deme

    @property
    def all_demes(self) -> Generator[Tuple[int, AbstractDeme], None, None]:
        for level_no in range(self.height):
            for deme in self.levels[level_no]:
                yield level_no, deme

    def demes(self, level_numbers) -> Generator[Tuple[int, AbstractDeme], None, None]:
        for level_no in level_numbers:
            for deme in self.levels[level_no]:
                yield level_no, deme

    @property
    def optima(self):
        return [leaf.best for leaf in self.leaves]

class DemeTree(AbstractDemeTree):
    def __init__(self, config: TreeConfig) -> None:
    
        super().__init__(0)
        nlevels = len(config.levels)
        if nlevels < 1:
            raise ValueError("Level number must be positive")

        self._config = config
        self._levels: List[List[Deme]] = [[] for _ in range(nlevels)]
        root_deme = Deme("root", config.levels[0], leaf=(nlevels == 1))
        self._levels[0].append(root_deme)

        self._gsc = config.gsc
        self._can_sprout = config.sprout_cond

    @property
    def levels(self):
        return self._levels

    @property
    def active_demes(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height):
            for deme in self.levels[level_no]:
                if deme.active:
                    yield level_no, deme

    @property
    def active_non_leaves(self) -> Generator[Tuple[int, Deme], None, None]:
        for level_no in range(self.height - 1):
            for deme in self.levels[level_no]:
                if deme.active:
                    yield level_no, deme

    def run(self):
        while not self._gsc(self):
            self._metaepoch_count += 1
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
            else:
                tree_logger.debug(f"Sprout refused for {deme}")

    def _do_sprout(self, deme, level):
        new_id = self._next_child_id(deme, level)
        is_leaf = (level == self.height - 1)
        child = Deme(
            id=new_id, 
            config=self._config.levels[level + 1], 
            started_at=self.metaepoch_count, 
            leaf=is_leaf,
            seed=max(deme.population)
            )
        deme.add_child(child)
        self._levels[level + 1].append(child)


    def _next_child_id(self, deme: Deme, level: int) -> str:
        if level >= self.height - 1:
            raise ValueError("Only non-leaf levels are admissible")

        id_suffix = len(self._levels[level + 1])
        if deme.id == "root":
            return str(id_suffix)
        else:
            return f"{deme.id}/{id_suffix}"
