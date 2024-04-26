from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.sprout.sprout_candidates import DemeCandidates
from pyhms.sprout.sprout_filters import (
    DemeLevelCandidatesFilter,
    DemeLimit,
    FarEnough,
    LevelLimit,
    NBC_FarEnough,
    TreeLevelCandidatesFilter,
)
from pyhms.sprout.sprout_generators import BestPerDeme, NBC_Generator, SproutCandidatesGenerator


class SproutMechanism:
    def __init__(
        self,
        candidates_generator: SproutCandidatesGenerator,
        deme_filter_chain: list[DemeLevelCandidatesFilter],
        tree_filter_chain: list[TreeLevelCandidatesFilter],
    ) -> None:
        super().__init__()
        self.candidates_generator = candidates_generator
        self.deme_filter_chain = deme_filter_chain
        self.tree_filter_chain = tree_filter_chain

    def get_seeds(self, tree) -> dict[AbstractDeme, DemeCandidates]:
        candidates = self.candidates_generator(tree)
        candidates = self.apply_deme_filters(candidates, tree)
        candidates = self.apply_tree_filters(candidates, tree)
        return {k: v for k, v in candidates.items() if len(candidates[k].individuals) > 0}

    def apply_deme_filters(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for filter in self.deme_filter_chain:
            candidates = filter(candidates, tree)
        return candidates

    def apply_tree_filters(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for filter in self.tree_filter_chain:
            candidates = filter(candidates, tree)
        return candidates


def get_NBC_sprout(
    gen_dist_factor: float = 3.0, trunc_factor: float = 0.7, fil_dist_factor: float = 3.0, level_limit: int = 4
) -> SproutMechanism:
    return SproutMechanism(
        NBC_Generator(gen_dist_factor, trunc_factor),
        [NBC_FarEnough(fil_dist_factor, 2), DemeLimit(1)],
        [LevelLimit(level_limit)],
    )


def get_simple_sprout(far_enough: float, level_limit: int = 4) -> SproutMechanism:
    return SproutMechanism(BestPerDeme(), [FarEnough(far_enough, 2)], [LevelLimit(level_limit)])
