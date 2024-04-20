from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import numpy.linalg as nla
from pyhms.demes.abstract_deme import AbstractDeme

from ..core.individual import Individual


class DemeLevelCandidatesFilter(ABC):
    @abstractmethod
    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        raise NotImplementedError()


class TreeLevelCandidatesFilter(ABC):
    @abstractmethod
    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        raise NotImplementedError()


class FarEnough(DemeLevelCandidatesFilter):
    def __init__(self, min_distance: float, norm_ord: int = 2) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.norm_ord = norm_ord

    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for deme in candidates.keys():
            child_siblings = [sibling for sibling in tree.levels[deme.level + 1] if sibling.is_active]
            child_seeds = candidates[deme][1]
            for sibling in child_siblings:
                child_seeds = list(
                    filter(
                        lambda ind: nla.norm(ind.genome - sibling.centroid, ord=self.norm_ord) > self.min_distance,
                        child_seeds,
                    )
                )
        return candidates


class NBC_FarEnough(DemeLevelCandidatesFilter):
    def __init__(self, min_distance_factor: float = 2.0, norm_ord: int = 2) -> None:
        super().__init__()
        self.min_distance_factor = min_distance_factor
        self.norm_ord = norm_ord

    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        assert (
            "NBC_mean_distance" in next(iter(candidates.values()))[0]
        ), "NBC_FarEnough filter requires NBC_mean_distance feature in candidates added throuhg NBC_Generator"

        for deme in candidates.keys():
            child_siblings = [sibling for sibling in tree.levels[deme.level + 1] if sibling.is_active]
            child_seeds = candidates[deme][1]
            for sibling in child_siblings:
                child_seeds = list(
                    filter(
                        lambda ind: nla.norm(ind.genome - sibling.centroid, ord=self.norm_ord)
                        > self.min_distance_factor * candidates[deme][0]["NBC_mean_distance"],
                        child_seeds,
                    )
                )
                candidates[deme] = (candidates[deme][0], child_seeds)
        return candidates


class DemeLimit(DemeLevelCandidatesFilter):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], _
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for deme in candidates.keys():
            candidates[deme][1].sort(key=lambda ind: ind.fitness)
            if len(candidates[deme][1]) > self.limit:
                candidates[deme] = (candidates[deme][0], candidates[deme][1][: self.limit])
        return candidates


class LevelLimit(TreeLevelCandidatesFilter):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def __call__(
        self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for level in range(len(tree.levels[:-1])):
            level_demes = [deme for deme in candidates.keys() if deme.level == level]
            level_candidates = [candidate for deme in level_demes for candidate in candidates[deme][1]]
            level_candidates.sort(key=lambda ind: ind.fitness)
            currently_active_level_below = len([deme for deme in tree.levels[level + 1] if deme.is_active])
            if currently_active_level_below + len(level_candidates) > self.limit:
                cutoff = self.limit - currently_active_level_below
                fitness_cutoff = level_candidates[cutoff].fitness
                for deme in level_demes:
                    candidates[deme] = (
                        candidates[deme][0],
                        list(filter(lambda ind: ind.fitness < fitness_cutoff, candidates[deme][1])),
                    )
        return candidates


class SkipSameSprout(TreeLevelCandidatesFilter):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]],
        tree,
    ) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for deme in candidates.keys():
            if not deme.children:
                continue
            children_sprout_genomes = np.array(
                [child._sprout_seed.genome for level_deme in tree.levels[deme.level] for child in level_deme.children]
            )
            not_equal_candidate_sprouts = [
                ind
                for ind in candidates[deme][1]
                if not np.any(np.all(np.isclose(children_sprout_genomes, ind.genome), axis=1))
            ]
            candidates[deme] = (candidates[deme][0], not_equal_candidate_sprouts)
        return candidates
