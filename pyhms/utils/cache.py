import numpy as np


class NumpyCache:
    def __init__(self):
        self.cache: dict[bytes, float] = {}

    def get_key(self, x: np.ndarray) -> bytes:
        return x.tobytes()

    def get(self, x: np.ndarray) -> float | None:
        key = self.get_key(x)
        return self.cache.get(key, None)

    def set(self, x: np.ndarray, value: float) -> None:
        key = self.get_key(x)
        self.cache[key] = value
