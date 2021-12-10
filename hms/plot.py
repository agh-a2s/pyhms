import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import argparse
from typing import List, Union
from leap_ec.individual import Individual

from hms.persist import Solution, DemeTreeData
from hms.grid import Grid2DEvaluation
from hms.util import bounds_to_extent

DEFAULT_CMAP = "gnuplot"

def scatterplot(population: Union[List[Solution], List[Individual]]):
    x = []
    y = []
    z = []
    for ind in population:
        x.append(ind.genome[0])
        y.append(ind.genome[1])
        z.append(ind.fitness)

    ax = plt.subplot()
    sct = ax.scatter(x, y, c=z, cmap=DEFAULT_CMAP)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(sct, cax=cax)
    return ax

def imageplot(grid: Grid2DEvaluation):
    ax = plt.subplot()
    ims = ax.imshow(grid.z.T, cmap=DEFAULT_CMAP, extent=bounds_to_extent(grid.bounds))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(ims, cax=cax)
    return ax

def main():
    args = parse_args()
    if args.file.endswith(".dat"):
        tree = DemeTreeData.load_binary(args.file)
        scatterplot(tree.level_individuals(args.level))
    elif args.file.endswith(".gdat"):
        grid = Grid2DEvaluation.load_binary(args.file)
        imageplot(grid)
    else:
        sys.exit("Unknown file extension")

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting contents of data file")
    parser.add_argument("file", help="data file")
    parser.add_argument("-l", "--level", help="tree level", type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main()