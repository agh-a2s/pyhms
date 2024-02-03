import numpy as np
import numpy.random as nrand


def sample_normal(
    center: np.array,
    std_dev: float,
    bounds: list[tuple[float, float]] | np.ndarray | None = None,
):
    """
    Sample points from a multivariate normal distribution.

    Args:
    - center (np.array): The mean of the distribution.
    - std_dev (float): the standard deviation for each dimension of the distribution.
        The covariance matrix is assumed to be diagonal, with each diagonal
        element being std_dev**2, indicating identical variance for each dimension
        and no covariance between dimensions.
    - bounds (list of tuples or np.array or None): Min and max bounds for each dimension.

    Returns a function that creates a sample from the distribution.
    """

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
