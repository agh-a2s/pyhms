"""
    Deme tree data.
"""
import pickle
from typing import List
from datetime import datetime

from .deme import DemeData
from ..tree import AbstractDemeTree, DemeTree

FILE_NAME_EXT = ".dat"

class DemeTreeData(AbstractDemeTree):
    def __init__(self, tree: DemeTree) -> None:
        super().__init__(tree.metaepoch_count)
        self._levels = [[] for _ in range(tree.height)]
        for lvl in range(tree.height):
            for deme in tree.level(lvl):
                self._levels[lvl].append(DemeData(deme))

    @property
    def levels(self) -> List[List[DemeData]]:
        return self._levels

    def save_binary(self, file_name_prefix="hms"):
        file_name = self.__class__._create_file_name(file_name_prefix)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def _create_file_name(prefix):
        dt_now = datetime.now()
        dt_part = dt_now.strftime('-%Y%m%d-%H%M%S')
        return prefix + dt_part + FILE_NAME_EXT

    @staticmethod
    def load_binary(file_name):
        out_tree = None
        with open(file_name, "rb") as infile:
            out_tree = pickle.load(infile)
        return out_tree
