import argparse
from leap_ec.problem import FunctionProblem

from hms.problem import StatsGatheringProblem, square
from hms.grid import Grid2DEvaluation

problem = StatsGatheringProblem(FunctionProblem(square, maximize=False))
bounds = [(-20, 20), (-20, 20)]

def main():
    args = parse_args()
    ge = Grid2DEvaluation(problem, bounds, granularity=args.granularity)
    ge.evaluate()
    ge.save_binary("square")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on 2D regular grid")
    parser.add_argument("-g" , "--granularity", type=float, default=0.1)
    return parser.parse_args()

if __name__ == "__main__":
    main()