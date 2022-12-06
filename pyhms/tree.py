from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

from .config import TreeConfig, EALevelConfig, CMALevelConfig
from .demes.abstract_deme import AbstractDeme
from .demes.ea_deme import EADeme
from .demes.cma_deme import CMADeme

class AbstractDemeTree(ABC):
    def __init__(self, metaepoch_count: int, config: TreeConfig) -> None:
        super().__init__()
        self._metaepoch_count = metaepoch_count
        self.config = config

    @property
    def metaepoch_count(self) -> int:
        return self._metaepoch_count

    @property
    @abstractmethod
    def levels(self) -> List[List[AbstractDeme]]:
        raise NotImplementedError()

    def level(self, no: int) -> List[AbstractDeme]:
        return self.levels[no]

    def level_individuals(self, level_no: int) -> list:
        inds = []
        for deme in self.level(level_no):
            inds += deme.all_individuals

        return inds

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
    
        super().__init__(0, config)
        nlevels = len(config.levels)
        if nlevels < 1:
            raise ValueError("Level number must be positive")

        self._levels: List[List[EADeme]] = [[] for _ in range(nlevels)]
        root_deme = EADeme("root", config.levels[0], leaf=(nlevels == 1))
        self._levels[0].append(root_deme)

        self._gsc = config.gsc
        self._can_sprout = config.sprout_cond

    @property
    def levels(self):
        return self._levels

    @property
    def active_demes(self) -> Generator[Tuple[int, EADeme], None, None]:
        for level_no in range(self.height):
            for deme in self.levels[level_no]:
                if deme.active:
                    yield level_no, deme

    @property
    def active_non_leaves(self) -> Generator[Tuple[int, EADeme], None, None]:
        for level_no in range(self.height - 1):
            for deme in self.levels[level_no]:
                if deme.active:
                    yield level_no, deme

    def run(self):
        while not self._gsc(self):
            self._metaepoch_count += 1
            self.run_metaepoch()
            if not self._gsc(self):
                self.run_sprout()

    def run_metaepoch(self):
        for level, deme in self.active_demes:
            deme.run_metaepoch()

    def run_sprout(self):
        for level, deme in self.active_non_leaves:
            if self._can_sprout(deme, level, self):
                self._do_sprout(deme, level)
            else:
                pass

    def _do_sprout(self, deme, level):
        new_id = self._next_child_id(deme, level)
        is_leaf = (level == self.height - 1)

        config = self.config.levels[level + 1]
        if isinstance(config, EALevelConfig):
            child = EADeme(
                id=new_id,
                config=config,
                started_at=self.metaepoch_count,
                leaf=is_leaf,
                seed=max(deme.population)
            )
        elif isinstance(config, CMALevelConfig):
            child = CMADeme(
                id=new_id,
                config=config,
                x0=deme.best,
                started_at=self.metaepoch_count
            )

        deme.add_child(child)
        self._levels[level + 1].append(child)


    def _next_child_id(self, deme: EADeme, level: int) -> str:
        if level >= self.height - 1:
            raise ValueError("Only non-leaf levels are admissible")

        id_suffix = len(self._levels[level + 1])
        if deme.id == "root":
            return str(id_suffix)
        else:
            return f"{deme.id}/{id_suffix}"
