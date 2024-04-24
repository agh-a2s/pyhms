from dataclasses import dataclass, field

from ..core.individual import Individual


@dataclass
class DemeCandidates:
    """Class for keeping an info about candidate solutions and ELA features data from a single deme."""
    individuals: list[Individual]
    features: dict[str, float | int] = field(default_factory=dict)