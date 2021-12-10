import argparse
import sys
from leap_ec.problem import FunctionProblem

from hms.experiments.erikkson.config_solver import erikkson, bounds
from hms.grid import Grid2DEvaluation

def main():
    args = parse_args()
    problem = erikkson(args.accuracy)
    ge = Grid2DEvaluation(problem, bounds, granularity=args.granularity)
    ge.evaluate()
    ge.save_binary("erikkson")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on 2D regular grid")
    parser.add_argument(
        "-g" , "--granularity", 
        type=float, 
        default=0.1,
        help="grid granularity (default: 0.1)"
        )
    parser.add_argument(
        "-a", "--accuracy", 
        type=int, 
        help="accuracy level (default: 0)", 
        choices=range(5),
        default=0
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()