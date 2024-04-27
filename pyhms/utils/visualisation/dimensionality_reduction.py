from typing import Protocol

import numpy as np


class DimensionalityReducer(Protocol):
    def fit(self, X: np.ndarray) -> None:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


class NaiveDimensionalityReducer(DimensionalityReducer):
    def __init__(self, selected_dimensions: tuple[int, int] = (0, 1)) -> None:
        self.selected_dimensions = selected_dimensions

    def fit(self, X: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_dimensions]
