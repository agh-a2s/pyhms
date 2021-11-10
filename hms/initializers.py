import numpy as np
import numpy.random as nrand

def sample_normal(center: np.array, std_dev: float, bounds=None):

    def in_bounds(x: np.array) -> bool:
        if bounds is None:
            return True
        else:
            bnd_arr = np.array(bounds)
            return np.all(x >= bnd_arr[:, 0]) and np.all(x <= bnd_arr[:, 1])

    def sample() -> np.array:
        return nrand.multivariate_normal(center, std_dev**2 * np.eye(len(center)))

    def create() -> np.array:
        x = sample()
        while not in_bounds(x):
            x = sample()

        return x

    return create
