from .tree import AbstractDemeTree

class Summary:
    def __init__(self, tree: AbstractDemeTree):
        self._tree = tree

    @property
    def metaepoch_count(self) -> int:
        return self._tree.metaepoch_count

    @property
    def number_of_levels(self) -> int:
        return self._tree.height

    @property
    def local_optima(self) -> list:
        return self._tree.optima
