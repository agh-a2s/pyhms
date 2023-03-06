import unittest

from pyhms.config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import deme_per_level_limit
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from leap_ec.problem import FunctionProblem


class TestSprout(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum(x**2)
    
    def test_deme_per_level_limit_sprout(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=10)
        sprout_cond = deme_per_level_limit(2)

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
            generations=10, 
            problem=function_problem, 
            bounds=[(-20, 20), (-20, 20)],
            sigma0=2.5,
            lsc=dont_stop()
            )
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nLevel limit sprout test")
        print("Deme info:")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")

        self.assertEqual(all([len(list(filter(lambda deme: deme.active, level))) <= 2 for level in tree.levels]), True, "Should be no more than 2 active demes on each level")
