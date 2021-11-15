"""
    Representation of a solution (point and fitness/objective).
"""
from typing import List
from leap_ec.individual import Individual
from leap_ec.problem import Problem

class Solution:
    def __init__(self, point, fitness: float, problem: Problem) -> None:
        self.point = point
        self.fitness = fitness
        self.problem = problem

    @classmethod
    def simplify(cls, individual: Individual):
        return cls(individual.genome, individual.fitness, individual.problem)

    @staticmethod
    def simplify_population(population: List[Individual]):
        return [Solution.simplify(ind) for ind in population]

    def __lt__(self, other):
        """
        a < b means a IS WORSE THAN b. In minimization context it means:
        a.fitness > b.fitness.
        """
        if other is None:
            return True

        return self.problem.worse_than(self.fitness, other.fitness)
