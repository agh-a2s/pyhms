from scipy.stats import chi2
import numpy as np


def calculate_chi_squared_threshold(percentile: float, dimensions: int) -> float:
    """
    Calculate the threshold for the Mahalanobis distance using the chi-squared distribution.
    """
    if not 0 < percentile < 1:
        raise ValueError("Percentile must be between 0 and 1")

    return chi2.ppf(percentile, df=dimensions)


def mahalanobis_distance(
    x: np.ndarray, y: np.ndarray, covariance_matrix_inverse: np.ndarray
) -> float:
    diff = x - y
    return np.sqrt(diff @ covariance_matrix_inverse @ diff)
