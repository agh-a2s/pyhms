import unittest

from pyhms.core.deme_config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.core.sprout import far_enough, deme_per_level_limit, composite_condition
from pyhms.stop_conditions.gsc import fitness_treshold
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.core.problem import EvalCountingProblem


class TestBasic(unittest.TestCase):

    @staticmethod
    def square(x) -> float:
        return sum([(dim-5)**2 for dim in x])

    def test_square_optimization_ea(self):
        function_problem = EvalCountingProblem(lambda x: self.square(x), 2, -20, 20)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = far_enough(1.0)

        config = [
        EALevelConfig(
            generations=2, 
            problem=function_problem,
            pop_size=20,
            lsc=dont_stop()
            ),
        EALevelConfig(
            generations=50, 
            problem=function_problem,
            pop_size=20,
            sample_std_dev=1.0,
            lsc=dont_stop()
            )
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nES square optimization test")
        for level, deme in tree.all_demes:
            print(f"Level {level}, {deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")

        self.assertGreater(len(tree.level(1)), 0, "Should sprout child demes")
    
    def test_square_optimization_cma(self):
        function_problem = EvalCountingProblem(lambda x: self.square(x), 2, -20, 20)
        gsc = metaepoch_limit(limit=2)
        sprout_cond = far_enough(1.0)

        config = [
        EALevelConfig(
            generations=2, 
            problem=function_problem,
            pop_size=20,
            lsc=dont_stop()
            ),
        CMALevelConfig(
            generations=100, 
            problem=function_problem,
            sigma0=0.1,
            lsc=dont_stop()
            )
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)

        print("\nCMA square optimization test")
        for level, deme in tree.all_demes:
            print(f"Level {level}, {deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")

        self.assertGreater(len(tree.level(1)), 0, "Should sprout child demes")
    
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
        for level, deme in tree.all_demes:
            print(f"Level {level}, {deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")

        self.assertEqual(all([0 < len(level) <= 2 for level in tree.levels]), True, "There should be active demes on each level but no more than 2")
    
    def test_fitness_treshold_gsc(self):
        function_problem = EvalCountingProblem(lambda x: self.square(x), 2, -20, 20)
        gsc = fitness_treshold(1.0e-6)
        sprout_cond = composite_condition([deme_per_level_limit(2), far_enough(1.0)])

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

        print("\nFitness treshold gsc test")
        for level, deme in tree.all_demes:
            print(f"Level {level}, {deme}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")
            print(f"Average fitness in last population {deme.avg_fitness()}")

        self.assertLess(tree.tree_bestever.get("F"), gsc.treshold, "Algorithm should not stop before reaching the fitness treshold")