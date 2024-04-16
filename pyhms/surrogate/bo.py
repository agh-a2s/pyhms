import warnings
from time import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .protocol import Surrogate


def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10, y_max_params=None):
    """Find the maximum of the acquisition function.

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (10) random starting points.

    Parameters
    ----------
    ac : callable
        Acquisition function to use. Should accept an array of parameters `x`,
        an from sklearn.gaussian_process.GaussianProcessRegressor `gp` and the
        best current value `y_max` as parameters.

    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A gaussian process regressor modelling the target function based on
        previous observations.

    y_max : number
        Highest found value of the target function.

    bounds : np.ndarray
        Bounds of the search space. For `N` parameters this has shape
        `(N, 2)` with `[i, 0]` the lower bound of parameter `i` and
        `[i, 1]` the upper bound.

    random_state : np.random.RandomState
        A random state to sample from.

    n_warmup : int, default=10000
        Number of points to sample from the acquisition function as seeds
        before looking for a minimum.

    n_iter : int, default=10
        Points to run L-BFGS-B optimization from.

    y_max_params : np.array
        Function parameters that produced the maximum known value given by `y_max`.

    :param y_max_params:
        Function parameters that produced the maximum known value given by `y_max`.

    Returns
    -------
    Parameters maximizing the acquisition function.

    """

    def adjusted_ac(x):
        return -ac(x.reshape(-1, bounds.shape[0]), gp=gp, y_max=y_max)

    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0]))
    ys = -adjusted_ac(x_tries)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more thoroughly
    x_seeds = random_state.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(1 + n_iter + int(y_max_params is not None), bounds.shape[0]),
    )
    # Add the best candidate from the random sampling to the seeds so that the
    # optimization algorithm can try to walk up to that particular local maxima
    x_seeds[0] = x_max
    if y_max_params is not None:
        # Add the provided best sample to the seeds so that the optimization
        # algorithm is aware of it and will attempt to find its local maxima
        x_seeds[1] = y_max_params
    start = time()
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(adjusted_ac, x_try, bounds=bounds, method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -np.squeeze(res.fun) >= max_acq:
            x_max = res.x
            max_acq = -np.squeeze(res.fun)
    print(f"Optimization time: {time() - start}")
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction:
    """An object to compute the acquisition functions.

    Parameters
    ----------
    kind: {'ucb', 'ei', 'poi'}
        * 'ucb' stands for the Upper Confidence Bounds method
        * 'ei' is the Expected Improvement method
        * 'poi' is the Probability Of Improvement criterion.

    kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is
            the highest.

    kappa_decay: float, optional(default=1)
        `kappa` is multiplied by this factor every iteration.

    kappa_decay_delay: int, optional(default=0)
        Number of iterations that must have passed before applying the
        decay to `kappa`.

    xi: float, optional(default=0.0)
    """

    def __init__(self, kind="ucb", kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ["ucb", "ei", "poi"]:
            err = "The utility function " f"{kind} has not been implemented, " "please choose one of ucb, ei, or poi."
            raise NotImplementedError(err)
        self.kind = kind

    def update_params(self):
        """Update internal parameters."""
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        """Calculate acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.


        Returns
        -------
        Values of the acquisition function
        """
        if self.kind == "ucb":
            return self.ucb(x, gp, self.kappa)
        if self.kind == "ei":
            return self.ei(x, gp, y_max, self.xi)
        if self.kind == "poi":
            return self.poi(x, gp, y_max, self.xi)
        raise ValueError(f"{self.kind} is not a valid acquisition function.")

    @staticmethod
    def ucb(x, gp, kappa):
        r"""Calculate Upper Confidence Bound acquisition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
        magnitude of improvement.
        Calculated as

        .. math::
            \text{UCB}(x) = \mu(x) + \kappa \sigma(x)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        kappa : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def ei(x, gp, y_max, xi):
        r"""Calculate Expected Improvement acqusition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
        magnitude of improvement.
        Calculated as

        .. math::
            \text{EI}(x) = (\mu(x)-y_{\text{max}} - \xi) \Phi\left(
                \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)
                  + \sigma(x) \phi\left(
                    \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def poi(x, gp, y_max, xi):
        r"""Calculate Probability of Improvement acqusition function.

        Calculated as

        .. math:: \text{POI}(x) = \Phi\left( \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF of the normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.
        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)


class BOSurrogate(Surrogate):
    def __init__(self, bounds: np.ndarray):
        self.utility_function = UtilityFunction(kind="ucb", kappa=5)
        self._random_state = np.random.RandomState()
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )
        self.bounds = bounds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        start = time()
        self._gp.fit(X, y)
        print(f"GP fit time: {time() - start}")
        self.y_max = np.min(y)
        self.x_max = X[np.argmin(y)]
        return self

    def suggest(self) -> np.ndarray:
        start = time()
        suggested = acq_max(
            ac=self.utility_function.utility,
            gp=self._gp,
            y_max=self.y_max,
            bounds=self.bounds,
            y_max_params=self.x_max,
            random_state=self._random_state,
        )
        print(f"Suggest time: {time() - start}")
        print(f"Suggested: {suggested}")
        return suggested
