import unittest

from pyhms.demes.deme_config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.core.sprout import far_enough
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from leap_ec.problem import FunctionProblem


class TestSquare(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum(x**2)

    def test_square_optimization_ea(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = far_enough(1.0)

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

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nES square optimization test")
        print("Deme info:")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")

        self.assertEqual(tree.height, 2, "Should be 2")
    
    def test_square_optimization_cma(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = far_enough(1.0)

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

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nCMA square optimization test")
        print("Deme info:")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")

        self.assertEqual(tree.height, 2, "Should be 2")
