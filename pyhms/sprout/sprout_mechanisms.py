from typing import List, Tuple, Dict
from leap_ec.individual import Individual

from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.sprout.sprout_filters import DemeLevelCandidatesFilter, TreeLevelCandidatesFilter, FarEnough, NBC_FarEnough, DemeLimit, LevelLimit
from pyhms.sprout.sprout_generators import SproutCandidatesGenerator, BestPerDeme, NBC_Generator

class SproutMechanism():

    def __init__(self, candidates_generator: SproutCandidatesGenerator, deme_filter_chain: List[DemeLevelCandidatesFilter], tree_filter_chain: List[TreeLevelCandidatesFilter]) -> None:
        super().__init__()
        self.candidates_generator = candidates_generator
        self.deme_filter_chain = deme_filter_chain
        self.tree_filter_chain = tree_filter_chain

    def get_seeds(self, tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        candidates = self.candidates_generator(tree)
        candidates = self.apply_deme_filters(candidates, tree)
        candidates = self.apply_tree_filters(candidates, tree)
        return {k: v for k, v in candidates.items() if len(candidates[k][1]) > 0}

    def apply_deme_filters(self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for filter in self.deme_filter_chain:
            candidates = filter(candidates, tree)
        return candidates
    
    def apply_tree_filters(self, candidates: Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]], tree) -> Dict[AbstractDeme, Tuple[Dict[str, float], List[Individual]]]:
        for filter in self.tree_filter_chain:
            candidates = filter(candidates, tree)
        return candidates

def get_NBC_sprout() -> SproutMechanism:
    return SproutMechanism(NBC_Generator(2.0, 1.0), [NBC_FarEnough(2.0, 2), DemeLimit(1)], [LevelLimit(4)])

def get_simple_sprout(far_enough: float, level_limit: float = 4) -> SproutMechanism:
    return SproutMechanism(BestPerDeme(), [FarEnough(far_enough, 2)], [LevelLimit(level_limit)])