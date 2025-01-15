"""
Bivariate Joe Copula Estimation
-------------------------------
"""

# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st
import sklearn.utils.validation as skv

from skfolio.distribution.copula.bivariate._base import (
    BaseBivariateCopula,
    CopulaRotation,
)

# Joe copula with a theta of 1.0 is just the independence copula, so we chose a lower
# bound of 1.005. After 20, the copula is already imposing very high tail dependence
# closed to comonotonic and increasing it will have negligible impact.
_THETA_BOUNDS = (1.005, 20.0)
_EULER_GAMMA = 0.5772156649015328606


class JoeCopula(BaseBivariateCopula):
    r"""Bivariate Joe Copula Estimation.

    The Joe copula is an Archimedean copula characterized by strong upper tail
    dependence and little to no lower tail dependence.

    It is used to Modeling extreme positive co-movements (simultaneous gains)
    By applying a 180° rotation, it can also be used for capturing simultaneous losses.

    It is defined by:

    .. math::
            C_{\theta}(u, v) = 1
            -
            \Bigl[
                (1 - u)^{\theta}
                \;+\;
                (1 - v)^{\theta}
                \;-\;
                (1 - u)^{\theta} (1 - v)^{\theta}
            \Bigr]^{\frac{1}{\theta}}

    where :math:`\theta \ge 1` is the dependence parameter. When :math:`\theta = 1`,
    the Joe copula reduces to the independence copula. Larger values of :math:`\theta`
    result in stronger upper-tail dependence.

    .. note::

        Rotation are needed for archimedean copulas (e.g., Joe, Gumbel, Clayton)
        because their parameters only model positive dependence, and they exhibit
        asymmetric tail behavior. To model negative dependence, one uses rotations
        (90°, 180°, or 270°) to “flip” the copula's tail dependence.


    Parameters
    ----------
    use_kendall_tau_inversion : bool, default=True
        Whether to use Kendall's tau inversion for estimating :math:`\theta`.

    kendall_tau : float, optional
        If `use_kendall_tau_inversion` is True and `kendall_tau` is provided, this
        value is used; otherwise, it is computed.

    rotation : CopulaRotation, optional
        The rotation to apply to the copula (default is no rotation).

    Attributes
    ----------
    theta_ : float
        Fitted theta coefficient :math:`\theta` > 1
    """

    theta_: float
    _n_params = 1

    def __init__(
        self,
        use_kendall_tau_inversion: bool = True,
        kendall_tau: float | None = None,
        rotation: CopulaRotation = CopulaRotation.R0,
    ):
        self.use_kendall_tau_inversion = use_kendall_tau_inversion
        self.kendall_tau = kendall_tau
        self.rotation = rotation

    def fit(self, X: npt.ArrayLike, y=None) -> "JoeCopula":
        """Fit the Bivariate Joe Copula.

         If `use_kendall_tau_inversion` is True, estimates :math:`\theta` using
         Kendall's tau inversion. Otherwise, uses MLE by maximizing the log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval [0, 1],
            having been transformed to uniform marginals.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        if self.use_kendall_tau_inversion:
            if self.kendall_tau is None:
                kendall_tau = st.kendalltau(X[:, 0], X[:, 1]).statistic
            else:
                kendall_tau = self.kendall_tau

            # Root-finding function brentq to find the value of theta in the interval
            # noinspection PyTypeChecker
            self.theta_ = so.brentq(
                _tau_diff, args=(kendall_tau,), a=_THETA_BOUNDS[0], b=_THETA_BOUNDS[-1]
            )

        else:
            result = so.minimize_scalar(
                _neg_log_likelihood, args=(X,), bounds=_THETA_BOUNDS, method="bounded"
            )
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            self.theta_ = result.x

        return self

    def partial_derivative(self, X: npt.ArrayLike) -> np.ndarray:
        r"""Compute the h-function (partial derivative) for the bivariate Joe copula.

        The h-function represents the conditional distribution function of :math:`u`
        given :math:`v`:

        .. math::
           h(u \mid v) = (1-u)^{\theta-1}\Bigl[1 - (1-v)^{\theta}\Bigr]
                         \Bigl[(1-u)^{\theta} + (1-v)^{\theta} - (1-u)^{\theta}(1-v)^{\theta}\Bigr]^{\frac{1}{\theta}-1}
                       = \left( 1 + \frac{(1-u)^{\theta}}{(1-v)^{\theta}} - (1-u)^{\theta} \right)^{-1 + \frac{1}{\theta}}
                       \cdot \left( 1 - (1-u)^{\theta} \right)

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        h : ndarray of shape (n_observations, )
            h-function values for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        x = np.power(1 - X[:, 0], self.theta_)
        y = np.power(1 - X[:, 1], self.theta_)
        h_values = np.power(1 + y / x - y, -1 + 1 / self.theta_) * (1.0 - y)
        return h_values

    def inverse_partial_derivative(self, X: npt.ArrayLike) -> np.ndarray:
        r"""Compute the inverse of the bivariate copula's partial derivative, commonly
        known as the inverse h-function [1]_.

        Let :math:`C(u, v)` be a bivariate copula. The h-function is defined by

        .. math::
            h(u \mid v) \;=\; \frac{\partial\,C(u, v)}{\partial\,v},

        which is the conditional distribution of :math:`U` given :math:`V = v`.
        The **inverse h-function**, denoted :math:`h^{-1}(p \mid v)`, is the unique
        value :math:`u \in [0,1]` such that

        .. math::
            h(u \mid v) \;=\; p,
            \quad \text{where } p \in [0,1].

        In practical terms, given :math:`(p, v)` in :math:`[0, 1]^2`,
        :math:`h^{-1}(p \mid v)` solves for the :math:`u` satisfying
        :math:`p = \partial C(u, v)/\partial v`.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(p, v)`, each in the interval `[0, 1]`.
            - The first column `p` corresponds to the value of the h-function.
            - The second column `v` is the conditioning variable.

        Returns
        -------
        u : ndarray of shape (n_observations, )
            A 1D-array of length `n_observations`, where each element is the computed
            :math:`u = h^{-1}(p \mid v)` for the corresponding pair in `X`.

        References
        ----------
        .. [1] "Multivariate Models and Dependence Concepts", Joe, H. (1997)
        .. [2] "An Introduction to Copulas", Nelsen, R. B. (2006)
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        p = X[:, 0]
        v = X[:, 1]

        x = np.power(1 - v, self.theta_)
        y = 1 - p * np.power(1 - v, 1 - self.theta_)
        y += x - y * x
        u = 1 - np.power(y, 1 / self.theta_)
        return u

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            The log-likelihood of each sample under the fitted copula.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        log_density = _sample_scores(X=X, theta=self.theta_)
        return log_density


def _neg_log_likelihood(theta: float, X: np.ndarray) -> float:
    """Negative log-likelihood function for optimization.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    theta : float
        Dependence parameter.

    Returns
    -------
    value : float
        The negative log-likelihood value.
    """
    return -np.sum(_sample_scores(X=X, theta=theta))


def _sample_scores(X: np.ndarray, theta: float) -> np.ndarray:
    """Compute the log-likelihood of each sample (log-pdf) under the bivariate
    Joe copula model.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

      theta : float
        Dependence parameter.

    Returns
    -------
    density : ndarray of shape (n_observations,)
        The log-likelihood of each sample under the fitted copula.

    Raises
    ------
    ValueError
        If rho is not in (-1, 1) or dof is not positive.
    """
    if theta <= 1.0:
        raise ValueError("Theta must be greater than 1 for the Joe copula.")

    u = X[:, 0]
    v = X[:, 1]

    # log-space transformation to improve stability (avoid overflow)
    x = np.log(1.0 - u)
    y = np.log(1.0 - v)
    x_y = x + y
    d = np.exp(x * theta) + np.exp(y * theta) - np.exp(x_y * theta)
    log_density = (
        (1.0 / theta - 2.0) * np.log(d) + x_y * (theta - 1.0) + np.log(theta - 1.0 + d)
    )
    return log_density


def _tau_diff(theta: float, tau_empirical: float) -> float:
    r"""Compute the difference between the theoretical Kendall's tau for the Joe copula
    and an empirical tau.

    The theoretical relationship for the Joe copula is given by:

    .. math::
       \tau(\theta) = 1 + \frac{2}{2-\theta} \left[ (1-\gamma) - \psi\left(\frac{2}{\theta}+1\right) \right],

    where :math:`\psi` is the digamma function and :math:`\gamma` is the Euler-Mascheroni constant.

    Parameters
    ----------
    theta : float
        The dependence parameter (must be greater than 1).

    tau_empirical : float
        The empirical Kendall's tau.

    Returns
    -------
    float
        The difference :math:`\tau(\theta) - \tau_{\text{empirical}}`.
    """
    # Euler-Mascheroni constant: gamma_const = 1 - EulerGamma
    gamma_const = 1.0 - _EULER_GAMMA
    # Compute theoretical tau using the digamma-based expression
    tau_theoretical = 1.0 + (2.0 / (2.0 - theta)) * (
        gamma_const - sp.digamma(2.0 / theta + 1.0)
    )
    return tau_theoretical - tau_empirical
