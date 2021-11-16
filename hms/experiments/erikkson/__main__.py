import logging

from ...algorithm import hms
from ...persist.tree import DemeTreeData
from .config import hms_config, gsc

logging.basicConfig(level=logging.DEBUG)

def main():
    tree = hms(level_config=hms_config, gsc=gsc)
    DemeTreeData(tree).save_binary("erikkson")

if __name__ == '__main__':
    main()