"""
    Deme tree data.
"""
import pickle
from typing import List
from datetime import datetime

from .pers_deme import DemeData
from ..tree import AbstractDemeTree, DemeTree
from ..utils.misc_util import unique_file_name

FILE_NAME_EXT = ".dat"

class DemeTreeData(AbstractDemeTree):
    def __init__(self, tree: DemeTree) -> None:
        super().__init__(tree.metaepoch_count, tree.config)
        self._levels = [[] for _ in range(tree.height)]
        for lvl in range(tree.height):
            for deme in tree.level(lvl):
                self._levels[lvl].append(DemeData(deme))

    @property
    def levels(self) -> List[List[DemeData]]:
        return self._levels

    def save_binary(self, file_name_prefix="hms"):
        file_name = unique_file_name(file_name_prefix, FILE_NAME_EXT)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_binary(file_name):
        out_tree = None
        with open(file_name, "rb") as infile:
            out_tree = pickle.load(infile)
        return out_tree
