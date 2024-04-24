from dataclasses import dataclass

import numpy as np

from ..core.individual import Individual


@dataclass
class DemeFeatures:
    """Class containing the numerical information calculated from the deme state before sprouting.

    Current list of features:
    NBC_mean_distance: Mean distance between individuals and their nearest better neighbor.
    """

    nbc_mean_distance: float | np.float64 | None = None


@dataclass
class DemeCandidates:
    """Class for keeping an info about candidate solutions and ELA features data from a single deme."""

    individuals: list[Individual]
    features: DemeFeatures
