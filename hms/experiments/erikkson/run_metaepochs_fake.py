import logging
import argparse
import sys

from hms.experiments.erikkson.config_fake import erikkson, bounds
from hms.config import EALevelConfig
from hms import hms
from hms.single_pop import SEA
from hms.stop_conditions.usc import dont_stop, metaepoch_limit
from hms.stop_conditions.lsc import fitness_steadiness
from hms.persist import DemeTreeData

hms_config = [
    EALevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=erikkson(0), 
        bounds=bounds, 
        pop_size=20,
        lsc=dont_stop()
        ),
    EALevelConfig(
        ea_class=SEA, 
        generations=2, 
        problem=erikkson(1), 
        bounds=bounds, 
        pop_size=5, 
        mutation_std=0.2, 
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
