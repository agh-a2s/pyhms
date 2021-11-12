"""
    Representation of a solution (point and fitness/objective).
"""
from typing import List
from leap_ec.individual import Individual

class Solution:
    def __init__(self, point, fitness: float) -> None:
        self.point = point
        self.fitness = fitness

    @classmethod
    def simplify(cls, individual: Individual):
        return cls(individual.genome, individual.fitness)

    @staticmethod
    def simplify_population(population: List[Individual]):
        return [Solution.simplify(ind) for ind in population]
        