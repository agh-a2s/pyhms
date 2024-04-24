from abc import ABC, abstractmethod

import numpy as np
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.sprout.sprout_candidates import DemeCandidates
from pyhms.utils.clusterization import NearestBetterClustering


class SproutCandidatesGenerator(ABC):
    @abstractmethod
    def __call__(self, tree) -> dict[AbstractDeme, DemeCandidates]:
        raise NotImplementedError()


class BestPerDeme(SproutCandidatesGenerator):
    def __call__(self, tree) -> dict[AbstractDeme, DemeCandidates]:
        return {
            deme: DemeCandidates(individuals=[deme.best_current_individual], features={})
            for level in tree.levels[:-1]
            for deme in level
            if deme.is_active
        }


class NBC_Generator(SproutCandidatesGenerator):
    def __init__(self, distance_factor: float, truncation_factor: float) -> None:
        self.distance_factor = distance_factor
        self.truncation_factor = truncation_factor
        super().__init__()

    def __call__(self, tree) -> dict[AbstractDeme, DemeCandidates]:
        candidates = {}
        for level in tree.levels[:-1]:
            for deme in level:
                if deme.is_active:
                    nbc = NearestBetterClustering(
                        deme.current_population,
                        self.distance_factor,
                        self.truncation_factor,
                    )
                    deme_candidate_inds = nbc.cluster()
                    candidates[deme] = DemeCandidates(
                        individuals=deme_candidate_inds,
                        features={"NBC_mean_distance": np.mean(nbc.distances)},
                    )
        return candidates  # type: ignore[return-value]


class NBCGeneratorWithLocalMethod(SproutCandidatesGenerator):
    def __init__(self, distance_factor: float, truncation_factor: float) -> None:
        self.distance_factor = distance_factor
        self.truncation_factor = truncation_factor
        super().__init__()

    def __call__(self, tree) -> dict[AbstractDeme, DemeCandidates]:
        candidates = {}
        for level in tree.levels[:-2]:
            for deme in level:
                if deme.is_active:
                    nbc = NearestBetterClustering(
                        deme.current_population,
                        self.distance_factor,
                        self.truncation_factor,
                    )
                    deme_candidate_inds = nbc.cluster()
                    candidates[deme] = DemeCandidates(
                        individuals=deme_candidate_inds,
                        features={"NBC_mean_distance": np.mean(nbc.distances)},
                    )
        for deme in tree.levels[-2]:
            if not deme.is_active and deme.started_at + len(deme.history) == tree.metaepoch_count:
                candidates[deme] = DemeCandidates(individuals=[deme.best_individual], features={})
        return candidates  # type: ignore[return-value]
