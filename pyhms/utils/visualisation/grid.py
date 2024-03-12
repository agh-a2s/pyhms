import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from leap_ec import Individual
from mpl_toolkits.axes_grid1 import make_axes_locatable

FILE_NAME_EXT = ".pkl"
DEFAULT_CMAP = "gnuplot"


def unique_file_name(prefix: str, ext: str = FILE_NAME_EXT) -> str:
    dt_part = datetime.now().strftime("-%Y%m%d-%H%M%S")
    return prefix + dt_part + ext


class Grid2DProblemEvaluation:
    def __init__(self, problem, bounds, granularity=0.1) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = bounds
        self.granularity = granularity
        self.z: np.ndarray

    def pickle_dump(self, filepath: str | None = None) -> None:
        if filepath is None:
            file_name_prefix = self.problem.__class__.__name__
            filepath = unique_file_name(file_name_prefix, FILE_NAME_EXT)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def pickle_load(filepath: str) -> "Grid2DProblemEvaluation":
        with open(filepath, "rb") as file:
            grid = pickle.load(file)
        return grid

    def evaluate(self) -> np.ndarray:
        xs = np.arange(self.bounds[0][0], self.bounds[0][1] + self.granularity, self.granularity)
        ys = np.arange(self.bounds[1][0], self.bounds[1][1] + self.granularity, self.granularity)
        self.z = np.zeros((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                self.z[i, j] = self.problem.evaluate(phenome=np.asarray([x, y]))
        return self.z

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

    def plot(self) -> None:
        ax = plt.subplot()
        extent = (
            self.bounds[0][0],
            self.bounds[0][1],
            self.bounds[1][0],
            self.bounds[1][1],
        )
        ims = ax.imshow(self.imshow_view, cmap=DEFAULT_CMAP, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ims, cax=cax)
        plt.show()


def plot_population(population: list[Individual]) -> None:
    x = [ind.genome[0] for ind in population]
    y = [ind.genome[1] for ind in population]
    z = [ind.fitness for ind in population]
    ax = plt.subplot()
    sct = ax.scatter(x, y, c=z, cmap=DEFAULT_CMAP)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sct, cax=cax)
    plt.show()
