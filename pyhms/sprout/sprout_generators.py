from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.utils.clusterization import NearestBetterClustering

from ..core.individual import Individual

# Would be nice to have a type alias for this. Although it requires python 3.12
# type SproutCandidates = Dict[AbstractDeme: (Dict[str: float], List[Individual])]


class SproutCandidatesGenerator(ABC):
    @abstractmethod
    def __call__(self, tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        raise NotImplementedError()


class BestPerDeme(SproutCandidatesGenerator):
    def __call__(self, tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        return {
            deme: ({}, [deme.best_current_individual]) for level in tree.levels[:-1] for deme in level if deme.is_active
        }


class NBC_Generator(SproutCandidatesGenerator):
    def __init__(self, distance_factor: float, truncation_factor: float) -> None:
        self.distance_factor = distance_factor
        self.truncation_factor = truncation_factor
        super().__init__()

    def __call__(self, tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        candidates = {}
        for level in tree.levels[:-1]:
            for deme in level:
                if deme.is_active:
                    nbc = NearestBetterClustering(
                        deme.current_population,
                        self.distance_factor,
                        self.truncation_factor,
                    )
                    deme_candidates = nbc.cluster()
                    candidates[deme] = (
                        {"NBC_mean_distance": np.mean(nbc.distances)},
                        deme_candidates,
                    )
        return candidates  # type: ignore[return-value]


class NBCGeneratorWithLocalMethod(SproutCandidatesGenerator):
    def __init__(self, distance_factor: float, truncation_factor: float) -> None:
        self.distance_factor = distance_factor
        self.truncation_factor = truncation_factor
        super().__init__()

    def __call__(self, tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        candidates = {}
        for level in tree.levels[:-2]:
            for deme in level:
                if deme.is_active:
                    nbc = NearestBetterClustering(
                        deme.current_population,
                        self.distance_factor,
                        self.truncation_factor,
                    )
                    deme_candidates = nbc.cluster()
                    candidates[deme] = (
                        {"NBC_mean_distance": np.mean(nbc.distances)},
                        deme_candidates,
                    )
        for deme in tree.levels[-2]:
            if not deme.is_active and deme.started_at + len(deme.history) == tree.metaepoch_count:
                candidates[deme] = ({}, [deme.best_individual])
        return candidates  # type: ignore[return-value]
