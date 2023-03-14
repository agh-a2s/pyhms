import unittest

from pyhms.demes.deme_config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.core.sprout import deme_per_level_limit
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.core.problem import EvalCountingProblem


class TestSprout(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum(x**2)
    
    def test_deme_per_level_limit_sprout(self):
        function_problem = EvalCountingProblem(lambda x: self.square(x), 2, -20, 20)
        gsc = metaepoch_limit(limit=10)
        sprout_cond = deme_per_level_limit(2)

        config = [
        EALevelConfig(
            generations=2, 
            problem=function_problem,
            pop_size=20,
            lsc=dont_stop()
            ),
        CMALevelConfig(
            generations=5, 
            problem=function_problem,
            sigma0=0.1,
            lsc=dont_stop()
            )
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nLevel limit sprout test")
        print("Deme info:")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")

        self.assertEqual(all([0 < len(level) <= 2 for level in tree.levels]), True, "There should be active demes on each level but no more than 2")