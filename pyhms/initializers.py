import numpy as np
import numpy.random as nrand
from scipy.stats import norm


def sample_uniform(bounds: np.ndarray):
    """
    Sample points from a uniform distribution.

    :param np.array bounds: Min and max bounds for each dimension.
    It has to be in a form of a 2-dimensional NumPy array of shape (n, 2).
    :param int size: Sample size.

    :return: A sample from the distribution.
    """

    def create():
        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=len(bounds))

    return create


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
    :param bounds: Min and max bounds for each dimension. It can be a numpy array, or None.
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


def sample_trunc_normal(center: np.ndarray, std_dev: float, bounds: np.ndarray):
    """
    Sample points from a truncated multivariate normal distribution.

    :param np.array center: The mean of the distribution.
    :param float std_dev: The standard deviation for each dimension of the distribution.
        The covariance matrix is assumed to be diagonal, with each diagonal
        element being `std_dev**2`, indicating identical variance for each dimension
        and no covariance between dimensions.
    :param bounds: Min and max bounds for each dimension. It can be a numpy array, or None.
    :type bounds: np.array or None
    :return: A function that creates a sample from the distribution.
    :rtype: function
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    lower_cdf = norm.cdf((lower - center) / std_dev)
    upper_cdf = norm.cdf((upper - center) / std_dev)

    def sample():
        uniform_samples = np.random.uniform(low=lower_cdf, high=upper_cdf, size=len(lower))
        normal_samples = norm.ppf(uniform_samples, loc=center, scale=std_dev)
        return normal_samples[0]

    return sample
