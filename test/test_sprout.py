import unittest
from leap_ec.problem import FunctionProblem
import numpy as np

from pyhms.config import TreeConfig, CMALevelConfig, EALevelConfig
from pyhms.tree import DemeTree
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout.sprout_mechanisms import get_simple_sprout, get_NBC_sprout
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.utils.misc_util import compute_centroid


class TestSprout(unittest.TestCase):

    @staticmethod
    def egg_holder(x) -> float:
        return sum([-x[i]*np.sin(np.sqrt(np.abs(x[i]-x[i+1]-47))) - (x[i]-47)*np.sin(np.sqrt(np.abs(0.5*x[i]+x[i+1]+47)))  for i in range(len(x)-2)])
    
    def test_simple_sprout(self):
        for limit in [2, 4, 8]:
            correct_sprout = True
            function_problem = FunctionProblem(lambda x: self.egg_holder(x), maximize=False)
            gsc = metaepoch_limit(limit=20)
            sprout_cond = get_simple_sprout(10.0, limit)

            config = [
            EALevelConfig(
                ea_class=SEA, 
                generations=2, 
                problem=function_problem, 
                bounds=[(-512, 512), (-512, 512), (-512, 512)], 
                pop_size=20,
                mutation_std=50.0,
                lsc=dont_stop()
                ),
            CMALevelConfig(
                generations=2,
                problem=function_problem, 
                bounds=[(-512, 512), (-512, 512), (-512, 512)],
                sigma0=2.5,
                lsc=metaepoch_limit(limit=8)
                )
            ]

            config = TreeConfig(config, gsc, sprout_cond)
            hms_tree = DemeTree(config)
            while not hms_tree._gsc(hms_tree):
                hms_tree.metaepoch_count += 1
                hms_tree.run_metaepoch()
                if not hms_tree._gsc(hms_tree):
                    hms_tree.run_sprout()
                
                if not all([len(list(filter(lambda deme: deme.is_active, level))) <= limit for level in hms_tree.levels]):
                    correct_sprout = False
                    print(f"\nFailed level limit sprout test. Limit: {limit}")
                    print("Deme info:")
                    for level, deme in hms_tree.all_demes:
                        print(f"Level {level}")
                        print(f"Active({deme.is_active}) {deme}")
                        print(f"Centroid position at the start {compute_centroid(deme.history[0])}")
                        print(f"Current centroid position {compute_centroid(deme.current_population)}")
                    break

                if hms_tree.metaepoch_count == 20:
                    print(f"\nSuccesful level limit sprout test. Limit: {limit}")
                    print("Deme info:")
                    for level, deme in hms_tree.all_demes:
                        print(f"Level {level}")
                        print(f"Active({deme.is_active}) {deme}")
                        print(f"Centroid position at the start {compute_centroid(deme.history[0])}")
                        print(f"Current centroid position {compute_centroid(deme.current_population)}")

            self.assertEqual(correct_sprout, True, "Should be no more than 2 active demes on each level")


    def test_default_nbc_sprout(self):
        correct_sprout = True
        function_problem = FunctionProblem(lambda x: self.egg_holder(x), maximize=False)
        gsc = metaepoch_limit(limit=20)
        sprout_cond = get_NBC_sprout()
        limit = 4

        config = [
        EALevelConfig(
            ea_class=SEA, 
            generations=2, 
            problem=function_problem, 
            bounds=[(-512, 512), (-512, 512), (-512, 512)], 
            pop_size=20,
            mutation_std=50.0,
            lsc=dont_stop()
            ),
        CMALevelConfig(
            generations=2,
            problem=function_problem, 
            bounds=[(-512, 512), (-512, 512), (-512, 512)],
            sigma0=2.5,
            lsc=metaepoch_limit(limit=8)
            )
        ]

        config = TreeConfig(config, gsc, sprout_cond)
        hms_tree = DemeTree(config)
        while not hms_tree._gsc(hms_tree):
            hms_tree.metaepoch_count += 1
            hms_tree.run_metaepoch()
            if not hms_tree._gsc(hms_tree):
                hms_tree.run_sprout()
            
            if not all([len(list(filter(lambda deme: deme.is_active, level))) <= limit for level in hms_tree.levels]):
                correct_sprout = False
                print(f"\nFailed default NBC sprout test. Limit: {limit}")
                print("Deme info:")
                for level, deme in hms_tree.all_demes:
                    print(f"Level {level}")
                    print(f"Active({deme.is_active}) {deme}")
                    print(f"Centroid position at the start {compute_centroid(deme.history[0])}")
                    print(f"Current centroid position {compute_centroid(deme.current_population)}")
                break

            if hms_tree.metaepoch_count == 19:
                print(f"\nSuccesful default NBC sprout test. Limit: {limit}")
                print("Deme info:")
                for level, deme in hms_tree.all_demes:
                    print(f"Level {level}")
                    print(f"Active({deme.is_active}) {deme}")
                    print(f"Centroid position at the start {compute_centroid(deme.history[0])}")
                    print(f"Current centroid position {compute_centroid(deme.current_population)}")

            self.assertEqual(correct_sprout, True, "Should be no more than 2 active demes on each level")
