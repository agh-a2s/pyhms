"""
    Deme data.
"""
from ..deme import AbstractDeme, Deme
from .solution import Solution

class DemeData(AbstractDeme):
    def __init__(self, deme: Deme) -> None:
        super().__init__(deme.id, deme._started_at)
        self._history = [Solution.simplify_population(pop) for pop in deme.history]

    @property
    def history(self) -> list:
        return self._history
