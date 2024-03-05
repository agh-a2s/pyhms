import numpy as np
from leap_ec import Individual
from pyhms.config import CMALevelConfig
from pyhms.demes.cma_deme import CMADeme
from pyhms.logging_ import get_logger
from pyhms.stop_conditions import DontStop
from pyhms.utils.print_tree import format_deme

from .config import SQUARE_BOUNDS, SQUARE_PROBLEM


def test_format_deme():
    logger = get_logger()
    config = CMALevelConfig(
        problem=SQUARE_PROBLEM,
        bounds=SQUARE_BOUNDS,
        lsc=DontStop(),
        sigma0=1.0,
        generations=1,
    )
    deme = CMADeme(
        id="0",
        level=0,
        config=config,
        logger=logger,
        x0=Individual(genome=np.array([0, 0]), problem=SQUARE_PROBLEM),
    )
    formatted_deme = format_deme(deme)
    assert formatted_deme.startswith("CMADeme 0")
    assert "sprout: (0.00, 0.00);" in formatted_deme
