import unittest

import numpy as np
from pyhms.demes.single_pop_eas.multiwinner import BlocPolicy, BordaPolicy, SNTVPolicy, get_positions_in_preferences


class TestVotingSchemes(unittest.TestCase):
    def test_bloc_voting(self):
        preferences = np.array([[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3], [2, 1, 3, 0]])
        k = 2
        target_winners = np.array([1, 2])
        voting = BlocPolicy()
        winners = voting(preferences, k)
        self.assertTrue(np.array_equal(winners, target_winners))

    def test_borda_voting(self):
        preferences = np.array([[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3], [2, 1, 3, 0]])
        k = 2
        target_winners = np.array([2, 1])
        voting = BordaPolicy()
        winners = voting(preferences, k)
        self.assertTrue(np.array_equal(winners, target_winners))

    def test_snt_voting(self):
        preferences = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 1, 3], [2, 1, 3, 0]])
        k = 2
        target_winners = np.array([0, 2])
        voting = SNTVPolicy()
        winners = voting(preferences, k)
        self.assertTrue(np.array_equal(winners, target_winners))

    def test_get_position_in_preferences(self):
        preferences = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [2, 1, 3, 0]])
        target_positions = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [3, 1, 0, 2]])
        positions = get_positions_in_preferences(preferences)
        self.assertTrue(np.array_equal(positions, target_positions))
