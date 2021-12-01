import numpy as np
import sys
import logging

from leap_ec.problem import FunctionProblem

from ...algorithm import local_optimization
from ...problem import StatsGatheringProblem, square

problem = StatsGatheringProblem(FunctionProblem(square, maximize=False))

bounds = [(-20, 20) for _ in range(2)]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

def main():
    x0 = np.asarray([float(sys.argv[1]), float(sys.argv[2])])
    logger.debug(f"x0 = {x0}")
    tree = local_optimization(x0=x0, problem=problem, bounds=bounds)
    sol = tree.optima[0]
    print(f"Point {sol.genome} value {sol.fitness}")

if __name__ == '__main__':
    main()