import unittest

from pyhms.config import CMALevelConfig, EALevelConfig
from pyhms.hms import hms
from pyhms.demes.single_pop_eas.sea import SEA
from pyhms.sprout import get_simple_sprout
from pyhms.stop_conditions.gsc import fitness_eval_limit_reached
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.problem import EvalCountingProblem, FunctionProblem


class TestIntegration(unittest.TestCase):
    @staticmethod
    def square(x) -> float:
        return sum(x**2)

    def test_hibernation_resume(self):
        function_problem = EvalCountingProblem(
            FunctionProblem(lambda x: self.square(x), maximize=False)
        )
        gsc = fitness_eval_limit_reached(limit=1000)
        sprout_cond = get_simple_sprout(1.0)
        options = {"hibernation": True}

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

        tree = hms(
            level_config=config, gsc=gsc, sprout_cond=sprout_cond, options=options
        )

        print("\nHibernating deme resume test")
        print(f"Root metaepoch count {len(tree.root.history)}")
        print("Deme info:")
        for level, deme in tree.all_demes:
            print(f"Level {level}")
            print(f"{deme}")

        self.assertGreater(
            len(tree.root.history),
            2,
            "Root deme should resume when it is possible to sprout",
        )
