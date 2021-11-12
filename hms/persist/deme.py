"""
    Deme data.
"""
from ..deme import Deme
from .solution import Solution

class DemeData:
    def __init__(self, deme: Deme) -> None:
        self.id = deme.id
        self.history = [Solution.simplify_population(pop) for pop in deme.history]
        self.started_at = deme._started_at
