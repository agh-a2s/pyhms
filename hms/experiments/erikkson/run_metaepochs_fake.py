import logging
import argparse
import sys

from .config_fake import erikkson, bounds

from ...config import LevelConfig
from ...algorithm import hms
from ...single_pop.sea import SEA
from ...usc import metaepoch_limit
from ...lsc import fitness_steadiness
from ...persist.tree import DemeTreeData

hms_config = [
    LevelConfig(SEA(2, erikkson(0), bounds, pop_size=20)),
    LevelConfig(
        SEA(2, erikkson(1), bounds, pop_size=5, mutation_std=0.2), 
        sample_std_dev=0.1, 
        lsc=fitness_steadiness(max_deviation=0.1)
        )
]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

def main():
    logger.debug(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metaepochs", type=int, default=20)
    args = parser.parse_args()
    if args.metaepochs <= 0:
        raise ValueError("Metaepoch number must be positive")
    logger.debug(f"Metaepoch limit {args.metaepochs}")

    gsc = metaepoch_limit(args.metaepochs)
    tree = hms(level_config=hms_config, gsc=gsc)
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()
