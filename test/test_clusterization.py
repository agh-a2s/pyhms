import unittest

from pyhms.utils.clusterization import NearestBetterClustering
from leap_ec.individual import Individual
from leap_ec.representation import Representation
from leap_ec.real_rep import create_real_vector
from leap_ec.problem import FunctionProblem

class TestSquare(unittest.TestCase):

    @staticmethod
    def two_squares(x) -> float:
        return min(sum(x**2), sum((x - [10.0]*len(x))**2))
    
    def test_nbc_tree_calculation(self):
        bounds=[(-20, 20)]*2
        function_problem = FunctionProblem(lambda x: self.two_squares(x), maximize=False)
        representation = Representation(initialize=create_real_vector(bounds=bounds))
        population = representation.create_population(pop_size=40, problem=function_problem)
        Individual.evaluate_population(population)

        clustering = NearestBetterClustering(population)
        clustering.prepare_spanning_tree()
        print(clustering.tree)

        self.assertTrue(True)