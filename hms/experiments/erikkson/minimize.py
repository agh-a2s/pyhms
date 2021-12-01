import numpy as np
import sys
import logging

from leap_ec.problem import FunctionProblem

from ...algorithm import local_optimization
from .config_solver import erikkson, bounds

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

def main():
    x0 = np.asarray([float(sys.argv[1]), float(sys.argv[2])])
    acc_level = int(sys.argv[3])
    logger.debug(f"x0 = {x0}")
    tree = local_optimization(x0=x0, problem=erikkson(acc_level), bounds=bounds)
    sol = tree.optima[0]
    print(f"Point {sol.genome} value {sol.fitness}")

if __name__ == '__main__':
    main()