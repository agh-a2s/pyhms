import numpy as np
import numpy.random as nrand


def sample_normal(
    center: np.ndarray,
    std_dev: float,
    bounds: np.ndarray | None = None,
):
    """
    Sample points from a multivariate normal distribution.

    :param np.array center: The mean of the distribution.
    :param float std_dev: The standard deviation for each dimension of the distribution.
        The covariance matrix is assumed to be diagonal, with each diagonal
        element being `std_dev**2`, indicating identical variance for each dimension
        and no covariance between dimensions.
    :param bounds: Min and max bounds for each dimension. It can be a list of tuples, a numpy array, or None.
    :type bounds: np.array or None
    :return: A function that creates a sample from the distribution.
    :rtype: function

    Example:

    .. code-block:: python

        >>> from pyhms.initializers import sample_normal
        >>> import numpy as np
        >>> bounds = [(-1, 1), (-1, 1)]
        >>> center = np.array([0, 0])
        >>> std_dev = 1.0
        >>> create_sample = sample_normal(center, std_dev, bounds)
        >>> sample = create_sample()
        >>> print(sample)
        [0.1 0.2]
    """

    def in_bounds(x: np.ndarray) -> np.bool_ | bool:
        if bounds is None:
            return True
        else:
            return np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1])

    def sample() -> np.ndarray:
        return nrand.multivariate_normal(center, std_dev**2 * np.eye(len(center)))

    def create() -> np.ndarray:
        x = sample()
        while not in_bounds(x):
            x = sample()

        return x

    return create
