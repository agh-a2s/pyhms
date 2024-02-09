import pathlib as pl
import os
import random
import unittest

import numpy as np
from pyhms.config import CMALevelConfig, DELevelConfig, EALevelConfig, TreeConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.tree import DemeTree

from .config import DEFAULT_GSC, DEFAULT_SPROUT_COND, SQUARE_PROBLEM, SQUARE_PROBLEM_DOMAIN, TEST_DIR


class TestPersistance(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.dump_file = os.path.join(TEST_DIR, 'hms_snapshot.pkl')

    def tearDown(self):
        os.remove(self.dump_file)

    def test_dumping(self):
        config = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND)
        hms_tree = DemeTree(config)
        hms_tree.run()
        hms_tree.pickle_dump(self.dump_file)
        self.assertTrue(pl.Path(self.dump_file).resolve().is_file())
        loaded_tree = DemeTree.pickle_load(self.dump_file)
        self.assertEqual(hms_tree.height, loaded_tree.height)
        self.assertEqual(hms_tree.best_individual.fitness, loaded_tree.best_individual.fitness)
        self.assertEqual(len(hms_tree.all_demes), len(loaded_tree.all_demes))
    
    def test_reload(self):
        options = {"log_level": "debug"}
        config = [
            DELevelConfig(
                generations=2,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                pop_size=20,
                dither=True,
                crossover=0.9,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=SQUARE_PROBLEM,
                bounds=SQUARE_PROBLEM_DOMAIN,
                sigma0=2.5,
                lsc=dont_stop(),
            ),
        ]

        config = TreeConfig(config, DEFAULT_GSC, DEFAULT_SPROUT_COND, options=options)
        hms_tree = DemeTree(config)
        hms_tree.run()
        hms_tree.pickle_dump(self.dump_file)
        loaded_tree = DemeTree.pickle_load(self.dump_file)
        self.assertEqual(loaded_tree.metaepoch_count, DEFAULT_GSC.limit)
        self.assertTrue(all(not deme._active for _, deme in loaded_tree.all_demes))
        loaded_tree._gsc = metaepoch_limit(limit=2*DEFAULT_GSC.limit)
        loaded_tree.root._active = True
        loaded_tree.levels[1][0]._active = True
        loaded_tree.run()
        self.assertEqual(loaded_tree.metaepoch_count, 2*DEFAULT_GSC.limit)
        self.assertGreater(len(loaded_tree.all_demes), len(hms_tree.all_demes))
        self.assertGreater(sum([len(deme.all_individuals) for _, deme in loaded_tree.all_demes]), sum([len(deme.all_individuals) for _, deme in hms_tree.all_demes]))
        self.assertLess(loaded_tree.best_individual.fitness, hms_tree.best_individual.fitness)
