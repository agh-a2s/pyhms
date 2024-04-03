from typing import Protocol

import numpy as np


class Surrogate(Protocol):
    def __init__(self, bounds: np.ndarray, **kwargs):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        pass

    def suggest(self) -> np.ndarray:
        pass
