import numpy as np
from pyhms.config import CMALevelConfig
from pyhms.core.individual import Individual
from pyhms.demes.abstract_deme import DemeInitArgs
from pyhms.demes.cma_deme import CMADeme
from pyhms.logging_ import get_logger
from pyhms.tree import DemeTree
from pyhms.utils.print_tree import format_deme

from .config import DEFAULT_LSC, SQUARE_PROBLEM, get_default_tree_config


def test_format_deme():
    logger = get_logger()
    config = CMALevelConfig(
        problem=SQUARE_PROBLEM,
        lsc=DEFAULT_LSC,
        sigma0=1.0,
        generations=1,
    )
    deme_init_args = DemeInitArgs(
        id="0",
        level=0,
        config=config,
        logger=logger,
        sprout_seed=Individual(genome=np.array([0, 0]), problem=SQUARE_PROBLEM),
    )
    deme = CMADeme(deme_init_args)
    formatted_deme = format_deme(deme)
    assert formatted_deme.startswith("CMADeme 0")
    assert "sprout: (0.00, 0.00);" in formatted_deme


def test_tree_and_summary():
    config = get_default_tree_config()
    tree = DemeTree(config)
    tree.run()
    formatted_tree = tree.tree()
    base_summary = tree.summary()
    short_summary = tree.summary(level_summary=False, deme_summary=False)
    assert base_summary.startswith(f"Metaepoch count: {tree.metaepoch_count}")
    assert short_summary.startswith(f"Metaepoch count: {tree.metaepoch_count}")
    assert formatted_tree in base_summary
    assert formatted_tree not in short_summary
