from dataclasses import dataclass, field

import numpy as np

from ..core.individual import Individual


@dataclass
class DemeCandidates:
    """Class for keeping an info about candidate solutions and ELA features data from a single deme."""

    individuals: list[Individual]
    features: dict[str, np.float64 | np.int64] = field(default_factory=dict)
