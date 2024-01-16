import unittest

from leap_ec.problem import FunctionProblem
from pyhms.config import CMALevelConfig, EALevelConfig
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.hms import hms
from pyhms.sprout import deme_per_level_limit
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit


class TestOptions(unittest.TestCase):
    @staticmethod
    def square(x) -> float:
        return sum(x**2)

    def test_local_optimization_ea(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=5)
        sprout_cond = deme_per_level_limit(1)

        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=function_problem,
                bounds=[(-20, 20), (-20, 20)],
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            EALevelConfig(
                ea_class=SEA,
                generations=4,
                problem=function_problem,
                bounds=[(-20, 20), (-20, 20)],
                pop_size=10,
                mutation_std=0.25,
                sample_std_dev=1.0,
                lsc=metaepoch_limit(4),
                run_minimize=True,
            ),
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)
        child = tree.root.children[0]

        print("\nLocal memetic optimization test for ea")
        print("Deme info:")
        print(f"One epoch before local optimization {max(child.history[-2])} and after {max(child.history[-1])}")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")

        self.assertGreaterEqual(
            max(child.history[-1]),
            max(child.history[-2]),
            "Quality after last metaepoch should be significantly better than before",
        )

    def test_local_optimization_cma(self):
        function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
        gsc = metaepoch_limit(limit=5)
        sprout_cond = deme_per_level_limit(1)

        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=function_problem,
                bounds=[(-20, 20), (-20, 20)],
                pop_size=20,
                mutation_std=1.0,
                lsc=dont_stop(),
            ),
            CMALevelConfig(
                generations=4,
                problem=function_problem,
                bounds=[(-20, 20), (-20, 20)],
                sigma0=2.5,
                lsc=metaepoch_limit(4),
                run_minimize=True,
            ),
        ]

        tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond)
        child = tree.root.children[0]

        print("\nLocal memetic optimization test for cma")
        print("Deme info:")
        print(f"One epoch before local optimization {max(child.history[-2])} and after {max(child.history[-1])}")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")
            print(f"Average fitness in last population {deme.avg_fitness()}")
            print(f"Average fitness in first population {deme.avg_fitness(0)}")

        self.assertGreaterEqual(
            max(child.history[-1]),
            max(child.history[-2]),
            "Quality after last metaepoch should be significantly better than before",
        )

    # def test_hibernation(self):
    #     function_problem = FunctionProblem(lambda x: self.square(x), maximize=False)
    #     gsc = metaepoch_limit(limit=10)
    #     sprout_cond = deme_per_level_limit(1)
    #     options = {'hibernation': True}

    #     config = [
    #     EALevelConfig(
    #         ea_class=SEA,
    #         generations=2,
    #         problem=function_problem,
    #         bounds=[(-20, 20), (-20, 20)],
    #         pop_size=20,
    #         mutation_std=1.0,
    #         lsc=dont_stop()
    #         ),
    #     CMALevelConfig(
    #         generations=4,
    #         problem=function_problem,
    #         bounds=[(-20, 20), (-20, 20)],
    #         sigma0=2.5,
    #         lsc=dont_stop()
    #         )
    #     ]

    #     tree = hms(level_config=config, gsc=gsc, sprout_cond=sprout_cond, options=options)

    #     print("\nHibernation mechanism test")
    #     print(f"Root metaepoch count {len(tree.root.history)} (First one for initialization and second one for singular run before sprouting)")
    #     print("Deme info:")
    #     for level, deme in tree.all_demes:
    #         print(f"Level {level}")
    #         print(f"{deme}")
    #         print(f"Average fitness in last population {deme.avg_fitness()}")
    #         print(f"Average fitness in first population {deme.avg_fitness(0)}")

    #     self.assertLess(len(tree.root.history), 3, "Root should hibernate while the other demes are still evolving")
