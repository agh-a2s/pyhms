import numpy as np
import pickle

from .util import unique_file_name

FILE_NAME_EXT = ".gdat"

class Grid2DEvaluation(object):
    def __init__(self, problem, bounds, granularity=0.1) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.granularity = granularity
        self.z = None

    def save_binary(self, file_name_prefix="grid"):
        file_name = unique_file_name(file_name_prefix, FILE_NAME_EXT)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_binary(file_name):
        grid = None
        with open(file_name, "rb") as infile:
            grid = pickle.load(infile)
        return grid

    def evaluate(self):
        xs = np.arange(
            self.bounds[0][0], 
            self.bounds[0][1] + self.granularity, 
            self.granularity
            )
        ys = np.arange(
            self.bounds[1][0], 
            self.bounds[1][1] + self.granularity, 
            self.granularity
            )
        self.z = np.zeros((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.z[i, j] = self.problem.evaluate(phenome=np.asarray([x, y]))

    def __str__(self) -> str:
        return f"Grid2DEvaluation(problem={self.problem}, bounds={self.bounds}, gran={self.granularity})"
