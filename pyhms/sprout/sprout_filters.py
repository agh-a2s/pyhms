from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as nla
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.sprout.sprout_candidates import DemeCandidates


class DemeLevelCandidatesFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        raise NotImplementedError()


class TreeLevelCandidatesFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        raise NotImplementedError()


class FarEnough(DemeLevelCandidatesFilter):
    def __init__(self, min_distance: float, norm_ord: int = 2) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.norm_ord = norm_ord

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for deme in candidates.keys():
            child_siblings = [sibling for sibling in tree.levels[deme.level + 1] if sibling.is_active]
            for sibling in child_siblings:
                candidates[deme].individuals = list(
                    filter(
                        lambda ind: nla.norm(ind.genome - sibling.centroid, ord=self.norm_ord) > self.min_distance,
                        candidates[deme].individuals,
                    )
                )
        return candidates


class NBC_FarEnough(DemeLevelCandidatesFilter):
    def __init__(self, min_distance_factor: float = 2.0, norm_ord: int = 2) -> None:
        super().__init__()
        self.min_distance_factor = min_distance_factor
        self.norm_ord = norm_ord

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        assert (
            "NBC_mean_distance" in next(iter(candidates.values())).features
        ), "NBC_FarEnough filter requires NBC_mean_distance feature in candidates added throuhg NBC_Generator"

        for deme in candidates.keys():
            child_siblings = [sibling for sibling in tree.levels[deme.level + 1] if sibling.is_active]
            for sibling in child_siblings:
                candidates[deme].individuals = list(
                    filter(
                        lambda ind: nla.norm(ind.genome - sibling.centroid, ord=self.norm_ord)
                        > self.min_distance_factor * candidates[deme].features["NBC_mean_distance"],
                        candidates[deme].individuals,
                    )
                )
        return candidates


class DemeLimit(DemeLevelCandidatesFilter):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        _,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for deme in candidates.keys():
            if len(candidates[deme].individuals) > self.limit:
                candidates[deme].individuals.sort(key=lambda ind: ind.fitness)
                candidates[deme].individuals = candidates[deme].individuals[: self.limit]
        return candidates


class LevelLimit(TreeLevelCandidatesFilter):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for level in range(len(tree.levels[:-1])):
            level_demes = [deme for deme in candidates.keys() if deme.level == level]
            level_candidates = [candidate for deme in level_demes for candidate in candidates[deme].individuals]
            level_candidates.sort(key=lambda ind: ind.fitness)
            currently_active_level_below = len([deme for deme in tree.levels[level + 1] if deme.is_active])
            if currently_active_level_below + len(level_candidates) > self.limit:
                cutoff = self.limit - currently_active_level_below
                fitness_cutoff = level_candidates[cutoff].fitness
                for deme in level_demes:
                    candidates[deme].individuals = list(
                        filter(lambda ind: ind.fitness < fitness_cutoff, candidates[deme].individuals)  # type: ignore
                    )
        return candidates


class SkipSameSprout(TreeLevelCandidatesFilter):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for deme in candidates.keys():
            if not deme.children:
                continue
            children_sprout_genomes = np.array(
                [child._sprout_seed.genome for level_deme in tree.levels[deme.level] for child in level_deme.children]
            )
            not_equal_candidate_sprouts = [
                ind
                for ind in candidates[deme].individuals
                if not np.any(np.all(np.isclose(children_sprout_genomes, ind.genome), axis=1))
            ]
            candidates[deme].individuals = not_equal_candidate_sprouts
        return candidates
