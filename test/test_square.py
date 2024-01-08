import unittest
from leap_ec.problem import FunctionProblem
import numpy as np

from pyhms.config import TreeConfig, CMALevelConfig, EALevelConfig, DELevelConfig
from pyhms.tree import DemeTree
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout.sprout_mechanisms import get_simple_sprout
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit


class TestSquare(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum(x**2)

    def test_square_optimization_ea(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = get_simple_sprout(1.0)

        config = [
        EALevelConfig(
            ea_class=SEA, 
            generations=2, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)], 
            pop_size=20,
            mutation_std=1.0,
            lsc=dont_stop()
            ),
        EALevelConfig(
            ea_class=SEA, 
            generations=4, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)], 
            pop_size=10,
            mutation_std=0.25,
            sample_std_dev=1.0,
            lsc=dont_stop()
            )
        ]

        config = TreeConfig(config, gsc, sprout_cond)
        hms_tree = DemeTree(config)
        while not hms_tree._gsc(hms_tree):
            hms_tree.metaepoch_count += 1
            hms_tree.run_metaepoch()
            if not hms_tree._gsc(hms_tree):
                hms_tree.run_sprout()

        print("\nES square optimization test")
        print("Deme info:")
        for level, deme in hms_tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {np.mean([ind.fitness for ind in deme.current_population])}")
            print(f"Average fitness in first population {np.mean([ind.fitness for ind in deme.history[0]])}")

        self.assertEqual(hms_tree.height, 2, "Should be 2")
    
    def test_square_optimization_cma(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = get_simple_sprout(1.0)

        config = [
        EALevelConfig(
            ea_class=SEA, 
            generations=2, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)], 
            pop_size=20,
            mutation_std=1.0,
            lsc=dont_stop()
            ),
        CMALevelConfig(
            generations=4, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)],
            sigma0=2.5,
            lsc=dont_stop()
            )
        ]

        config = TreeConfig(config, gsc, sprout_cond)
        hms_tree = DemeTree(config)
        while not hms_tree._gsc(hms_tree):
            hms_tree.metaepoch_count += 1
            hms_tree.run_metaepoch()
            if not hms_tree._gsc(hms_tree):
                hms_tree.run_sprout()

        print("\nCMA square optimization test")
        print("Deme info:")
        for level, deme in hms_tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {np.mean([ind.fitness for ind in deme.current_population])}")
            print(f"Average fitness in first population {np.mean([ind.fitness for ind in deme.history[0]])}")

        self.assertEqual(hms_tree.height, 2, "Should be 2")

    def test_square_optimization_de(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = get_simple_sprout(1.0)

        config = [
        DELevelConfig(
            generations=2, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)], 
            pop_size=20,
            dither=True,
            crossover=0.9,
            lsc=dont_stop()
            ),
        CMALevelConfig(
            generations=4, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)],
            sigma0=2.5,
            lsc=dont_stop()
            )
        ]

        config = TreeConfig(config, gsc, sprout_cond)
        hms_tree = DemeTree(config)
        while not hms_tree._gsc(hms_tree):
            hms_tree.metaepoch_count += 1
            hms_tree.run_metaepoch()
            if not hms_tree._gsc(hms_tree):
                hms_tree.run_sprout()

        print("\nDE square optimization test")
        print("Deme info:")
        for level, deme in hms_tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {np.mean([ind.fitness for ind in deme.current_population])}")
            print(f"Average fitness in first population {np.mean([ind.fitness for ind in deme.history[0]])}")

        self.assertEqual(hms_tree.height, 2, "Should be 2")
