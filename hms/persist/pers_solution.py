"""
    Representation of a solution (point and value).
"""
import numpy as np
from typing import List
from leap_ec.individual import Individual
from leap_ec.problem import Problem

class Solution:
    def __init__(self, point, value: float, problem: Problem) -> None:
        self.point = point
        self.value = value
        self.problem = problem

    @classmethod
    def simplify(cls, individual: Individual):
        return cls(individual.genome, individual.fitness, individual.problem)

    @staticmethod
    def simplify_population(population: List[Individual]):
        return [Solution.simplify(ind) for ind in population]

    @property
    def genome(self) -> np.array:
        return np.asarray(self.point)

    @property
    def fitness(self) -> float:
        return self.value

    def __lt__(self, other):
        """
        a < b means a IS WORSE THAN b. In minimization context it means:
        a.value > b.value.
        """
        if other is None:
            return True

        return self.problem.worse_than(self.value, other.value)

    def __str__(self) -> str:
        return f"Pt. {self.point} val. {self.value}"
