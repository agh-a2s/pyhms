import pickle
from datetime import datetime

import numpy as np

FILE_NAME_EXT = ".gdat"


def unique_file_name(prefix, ext):
    dt_now = datetime.now()
    dt_part = dt_now.strftime("-%Y%m%d-%H%M%S")
    return prefix + dt_part + ext


class Grid2DEvaluation(object):
    def __init__(self, problem, bounds, granularity=0.1) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.granularity = granularity
        self.z: np.ndarray

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
        xs = np.arange(self.bounds[0][0], self.bounds[0][1] + self.granularity, self.granularity)
        ys = np.arange(self.bounds[1][0], self.bounds[1][1] + self.granularity, self.granularity)
        self.z = np.zeros((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.z[i, j] = self.problem.evaluate(phenome=np.asarray([x, y]))

    @property
    def imshow_view(self) -> np.ndarray:
        return np.flipud(self.z.T)

    def __str__(self) -> str:
        return f"Grid2DEvaluation(problem={self.problem}, bounds={self.bounds}, gran={self.granularity})"

    def summary(self) -> str:
        s = "Grid 2D evaluation:\n"
        s += f"Problem: {self.problem}\n"
        s += f"Bounds: {self.bounds}\n"
        s += f"Granularity: {self.granularity}"
        return s
