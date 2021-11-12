"""
    Deme tree data.
"""
import pickle
import json

from .deme import DemeData
from ..tree import DemeTree

class DemeTreeData:
    def __init__(self, tree: DemeTree) -> None:
        self.meatapoch_count = tree.metaepoch_count
        self.levels = [[] for _ in range(tree.height)]
        for lvl in range(tree.height):
            for deme in tree.level(lvl):
                self.levels[lvl].append(DemeData(deme))

    def save_binary(self, file_name="hms.dat"):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_binary(file_name="hms.dat"):
        tree = None
        with open(file_name, "rb") as f:
            tree = pickle.load(f)

        return tree
