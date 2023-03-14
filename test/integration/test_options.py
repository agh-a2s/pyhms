import unittest

from pyhms.demes.deme_config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.core.sprout import deme_per_level_limit
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.core.problem import EvalCountingProblem


class TestOptions(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum(x**2)

    def test_local_optimization(self):
        function_problem = EvalCountingProblem(lambda x: self.square(x), 2, -20, 20)
        gsc = metaepoch_limit(limit=5)
        sprout_cond = deme_per_level_limit(1)

        config = [
        EALevelConfig(
            generations=2, 
            problem=function_problem,
            pop_size=20,
            lsc=dont_stop()
            ),
        EALevelConfig(
            generations=4, 
            problem=function_problem,
            pop_size=10,
            sample_std_dev=1.0,
            lsc=metaepoch_limit(4),
            run_minimize=True
            )
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)
        child = tree.root.children[0]

        print("\nLocal optimization test")
        print("Deme info:")
        print(f'One epoch before local optimization {min(child.history[-2], key=lambda x: x.get("F")).get("F")} and after {child.best.get("F")}')
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
 
        self.assertGreaterEqual(min(child.history[-2], key=lambda x: x.get("F")).get("F"), child.best.get("F"), "Quality after last metaepoch should be significantly better than before")
