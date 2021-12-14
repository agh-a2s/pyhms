import argparse
from leap_ec.problem import FunctionProblem

from hms.problem import StatsGatheringProblem
from hms.problems.relay2d import relay
from hms.grid import Grid2DEvaluation

problem = StatsGatheringProblem(FunctionProblem(relay, maximize=False))
bounds = [(-100, 100), (0, 100)]

def main():
    args = parse_args()
    ge = Grid2DEvaluation(problem, bounds, granularity=args.granularity)
    ge.evaluate()
    ge.save_binary("relay2d")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on 2D regular grid")
    parser.add_argument("-g" , "--granularity", type=float, default=0.1)
    return parser.parse_args()

if __name__ == "__main__":
    main()