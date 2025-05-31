"""Bivariate Joe Copula Estimation."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st
import sklearn.utils.validation as skv

from skfolio.distribution.copula._base import BaseBivariateCopula
from skfolio.distribution.copula._utils import (
    CopulaRotation,
    _apply_copula_rotation,
    _apply_margin_swap,
    _apply_rotation_cdf,
    _apply_rotation_partial_derivatives,
    _select_rotation_itau,
    _select_theta_and_rotation_mle,
)

# Joe copula with a theta of 1.0 is just the independence copula, so we chose a lower
# bound of 1.005. After 20, the copula is already imposing very high tail dependence
# closed to comonotonic and increasing it will make it impractical.
_THETA_BOUNDS = (1.005, 20.0)
_EULER_GAMMA = 0.5772156649015328606


class JoeCopula(BaseBivariateCopula):
    r"""Bivariate Joe Copula Estimation.

    The Joe copula is an Archimedean copula characterized by strong upper tail
    dependence and little to no lower tail dependence.

    In its unrotated form, it is used for modeling extreme co-movements in the upper
    tail (i.e. simultaneous extreme gains).

    Rotations allow the copula to be adapted for different types of tail dependence:
      - A 180° rotation captures extreme co-movements in the lower tail (i.e.
        simultaneous extreme losses).

      - A 90° rotation captures scenarios where one variable exhibits extreme losses
        while the other shows extreme gains.

      - A 270° rotation captures the opposite scenario, where one variable experiences
        extreme gains while the other suffers extreme losses.

    Joe copula generally exhibits stronger upper tail dependence than the Gumbel copula.

    It is defined by:

    .. math::
            C_{\theta}(u, v) = 1-\Bigl[(1 - u)^{\theta} + (1 - v)^{\theta} -
                (1 - u)^{\theta} (1 - v)^{\theta}\Bigr]^{\frac{1}{\theta}}

    where :math:`\theta \ge 1` is the dependence parameter. When :math:`\theta = 1`,
    the Joe copula reduces to the independence copula. Larger values of :math:`\theta`
    result in stronger upper-tail dependence.

    .. note::

        Rotation are needed for archimedean copulas (e.g., Joe, Gumbel, Clayton)
        because their parameters only model positive dependence, and they exhibit
        asymmetric tail behavior. To model negative dependence, one uses rotations
        to “flip” the copula's tail dependence.

    Parameters
    ----------
    itau : bool, default=True
        If True, :math:`\theta` is estimated using the Kendall's tau inversion method;
        otherwise, the Maximum Likelihood Estimation (MLE) method is used. The MLE is
        slower but more accurate.

    kendall_tau : float, optional
        If `itau` is True and `kendall_tau` is provided, this value is used;
        otherwise, it is computed.

    tolerance : float, default=1e-4
        Convergence tolerance for the MLE optimization.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Attributes
    ----------
    theta_ : float
        Fitted theta coefficient :math:`\theta` > 1.

    rotation_ : CopulaRotation
        Fitted rotation of the copula.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution import JoeCopula, compute_pseudo_observations
    >>>
    >>> # Load historical prices and convert them to returns
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> X = X[["AAPL", "JPM"]]
    >>>
    >>> # Convert returns to pseudo observation in the interval [0,1]
    >>> X = compute_pseudo_observations(X)
    >>>
    >>> # Initialize the Copula estimator
    >>> model = JoeCopula()
    >>>
    >>> # Fit the model to the data.
    >>> model.fit(X)
    >>>
    >>> # Display the fitted parameter and tail dependence coefficients
    >>> print(model.fitted_repr)
    JoeCopula(theta=1.48, rot=180°)
    >>> print(model.lower_tail_dependence)
    0.4021
    >>> print(model.upper_tail_dependence)
    0.0
    >>>
    >>> # Compute the log-likelihood, total log-likelihood, CDF, Partial Derivative,
    >>> # Inverse Partial Derivative, AIC, and BIC
    >>> log_likelihood = model.score_samples(X)
    >>> score = model.score(X)
    >>> cdf = model.cdf(X)
    >>> p = model.partial_derivative(X)
    >>> u = model.inverse_partial_derivative(X)
    >>> aic = model.aic(X)
    >>> bic = model.bic(X)
    >>>
    >>> # Generate 5 new samples
    >>> samples = model.sample(n_samples=5)
    >>>
    >>> # Plot the tail concentration function.
    >>> fig = model.plot_tail_concentration()
    >>> fig.show()
    >>>
    >>> # Plot a 2D contour of the estimated PDF.
    >>> fig = model.plot_pdf_2d()
    >>> fig.show()
    >>>
    >>> # Plot a 3D surface of the estimated PDF.
    >>> fig = model.plot_pdf_3d()
    >>> fig.show()

    References
    ----------
    .. [1] "An Introduction to Copulas (2nd ed.)",
       Nelsen (2006)

    .. [2] "Multivariate Models and Dependence Concepts",
        Joe, Chapman & Hall (1997)

    .. [3] "Quantitative Risk Management: Concepts, Techniques and Tools",
        McNeil, Frey & Embrechts (2005)

    .. [4] "The t Copula and Related Copulas",
        Demarta & McNeil (2005)

    .. [5] "Copula Methods in Finance",
        Cherubini, Luciano & Vecchiato (2004)
    """

    theta_: float
    rotation_: CopulaRotation
    _n_params = 1

    def __init__(
        self,
        itau: bool = True,
        kendall_tau: float | None = None,
        tolerance: float = 1e-4,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self.itau = itau
        self.kendall_tau = kendall_tau
        self.tolerance = tolerance

    def fit(self, X: npt.ArrayLike, y=None) -> "JoeCopula":
        r"""Fit the Bivariate Joe Copula.

        If `itau` is True, estimates :math:`\theta` using Kendall's tau inversion.
        Otherwise, uses MLE by maximizing the log-likelihood.

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

        if self.itau:
            if self.kendall_tau is None:
                kendall_tau = st.kendalltau(X[:, 0], X[:, 1]).statistic
            else:
                kendall_tau = self.kendall_tau

            abs_kendall_tau = abs(kendall_tau)

            # Root-finding function brentq to find the value of theta in the interval
            # brentq fails if _tau_diff has same sign, it happens when we are at the
            # bounds so we capture it before.
            fa = _tau_diff(_THETA_BOUNDS[0], abs_kendall_tau)
            fb = _tau_diff(_THETA_BOUNDS[1], abs_kendall_tau)
            if fa * fb > 0:
                if abs(fa) < abs(fb):
                    self.theta_ = _THETA_BOUNDS[0]
                else:
                    self.theta_ = _THETA_BOUNDS[1]
            else:
                # noinspection PyTypeChecker
                self.theta_ = so.brentq(
                    _tau_diff,
                    args=(abs_kendall_tau,),
                    a=_THETA_BOUNDS[0],
                    b=_THETA_BOUNDS[-1],
                )
            self.rotation_ = _select_rotation_itau(
                func=_neg_log_likelihood, X=X, theta=self.theta_
            )

        else:
            self.theta_, self.rotation_ = _select_theta_and_rotation_mle(
                _neg_log_likelihood, X=X, bounds=_THETA_BOUNDS, tolerance=self.tolerance
            )

        return self

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the CDF of the bivariate Joe copula.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        cdf : ndarray of shape (n_observations,)
            CDF values for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        cdf = _apply_rotation_cdf(
            func=_base_cdf, X=X, rotation=self.rotation_, theta=self.theta_
        )
        return cdf

    def partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the h-function (partial derivative) for the bivariate Joe copula
        with respect to a specified margin.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math::  \begin{aligned}
                   h(u \mid v)
                     &= \frac{\partial C(u,v)}{\partial v} \\[6pt]
                     &= (1-v)^{\theta-1}\,\Bigl[1 \;-\;(1-u)^{\theta}\Bigr]\,
                        \Bigl[(1-u)^{\theta} \;+\;(1-v)^{\theta}
                              \;-\;(1-u)^{\theta}(1-v)^{\theta}\Bigr]^{\frac{1}{\theta}-1} \\[6pt]
                     &= \left( 1 \;+\;\frac{(1-u)^{\theta}}{(1-v)^{\theta}}
                              \;-\;(1-u)^{\theta} \right)^{-1 + \frac{1}{\theta}}
                        \;\cdot\;\bigl[\,1 \;-\;(1-u)^{\theta}\bigr].
                   \end{aligned}

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        first_margin : bool, default=False
            If True, compute the partial derivative with respect to the first
            margin `u`; otherwise, compute the partial derivative with respect to the
            second margin `v`.

        Returns
        -------
        p  : ndarray of shape (n_observations,)
            h-function values :math:`h(u \mid v) \;=\; p` for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        p = _apply_rotation_partial_derivatives(
            func=_base_partial_derivative,
            X=X,
            rotation=self.rotation_,
            first_margin=first_margin,
            theta=self.theta_,
        )
        return p

    def inverse_partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the inverse of the bivariate copula's partial derivative, commonly
        known as the inverse h-function [1]_.

        Let :math:`C(u, v)` be a bivariate copula. The h-function with respect to the
        second margin is defined by

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

        first_margin : bool, default=False
            If True, compute the inverse partial derivative with respect to the first
            margin `u`; otherwise, compute the inverse partial derivative with respect
            to the second margin `v`.

        Returns
        -------
        u : ndarray of shape (n_observations,)
            A 1D-array of length `n_observations`, where each element is the computed
            :math:`u = h^{-1}(p \mid v)` for the corresponding pair in `X`.

        References
        ----------
        .. [1] "Multivariate Models and Dependence Concepts", Joe, H. (1997)
        .. [2] "An Introduction to Copulas", Nelsen, R. B. (2006)
        .. [3] . "Nested Archimedean Copulas Meet ", Hofert & Mächler (2011)
        """
        # no known closed-form solution, hence we use Newton method.
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        u = _apply_rotation_partial_derivatives(
            func=_base_inverse_partial_derivative,
            X=X,
            rotation=self.rotation_,
            first_margin=first_margin,
            theta=self.theta_,
        )
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
        X = _apply_copula_rotation(X, rotation=self.rotation_)
        log_density = _base_sample_scores(X=X, theta=self.theta_)
        return log_density

    @property
    def lower_tail_dependence(self) -> float:
        """Theoretical lower tail dependence coefficient."""
        skv.check_is_fitted(self)
        if self.rotation_ == CopulaRotation.R180:
            return 2.0 - np.power(2.0, 1.0 / self.theta_)
        return 0

    @property
    def upper_tail_dependence(self) -> float:
        """Theoretical upper tail dependence coefficient."""
        skv.check_is_fitted(self)
        if self.rotation_ == CopulaRotation.R0:
            return 2.0 - np.power(2.0, 1.0 / self.theta_)
        return 0

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted copula."""
        return (
            f"{self.__class__.__name__}(theta={self.theta_:0.2f}, rot={self.rotation_})"
        )


def _neg_log_likelihood(theta: float, X: np.ndarray) -> float:
    """Negative log-likelihood function for optimization.

    Parameters
    ----------
     X : array-like of shape (n_observations, 2)
         An array of bivariate inputs `(u, v)` where each row represents a
         bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
         having been transformed to uniform marginals.

    theta : float
         The dependence parameter (must be greater than 1).

    Returns
    -------
     value : float
         The negative log-likelihood value.
    """
    return -np.sum(_base_sample_scores(X=X, theta=theta))


def _base_sample_scores(X: np.ndarray, theta: float) -> np.ndarray:
    """Compute the log-likelihood of each sample (log-pdf) under the bivariate
    Joe copula model.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    theta : float
        The dependence parameter (must be greater than 1).

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

    # log-space transformation to improve stability near 0  or 1
    x, y = np.log1p(-X).T
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

    where :math:`\psi` is the digamma function and :math:`\gamma` is the
    Euler-Mascheroni constant.

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


def _base_cdf(X: np.ndarray, theta: float) -> np.ndarray:
    """Bivariate Joe CDF (unrotated)."""
    z = np.power(1 - X, theta)
    cdf = 1.0 - np.power(np.sum(z, axis=1) - np.prod(z, axis=1), 1.0 / theta)
    return cdf


def _base_partial_derivative(
    X: np.ndarray, first_margin: bool, theta: float
) -> np.ndarray:
    r"""Compute the h-function (partial derivative) for the bivariate unrotated
    Joe copula with respect to a specified margin.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    first_margin : bool, default=False
        If True, compute the partial derivative with respect to the first
        margin `u`; otherwise, compute the partial derivative with respect to the
        second margin `v`.

    theta : float
        The dependence parameter (must be greater than 1).

    Returns
    -------
      : ndarray of shape (n_observations,)
        h-function values :math:`h(u \mid v) \;=\; p` for each observation in X.
    """
    X = _apply_margin_swap(X, first_margin=first_margin)
    x, y = np.power(1 - X, theta).T
    p = np.power(1 + x / y - x, 1 / theta - 1) * (1.0 - x)
    return p


def _base_inverse_partial_derivative(
    X: np.ndarray, first_margin: bool, theta: float
) -> np.ndarray:
    r"""Compute the inverse of the bivariate copula's partial derivative, commonly
    known as the inverse h-function.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(p, v)`, each in the interval `[0, 1]`.
        - The first column `p` corresponds to the value of the h-function.
        - The second column `v` is the conditioning variable.

    first_margin : bool, default=False
        If True, compute the inverse partial derivative with respect to the first
        margin `u`; otherwise, compute the inverse partial derivative with respect to
        the second margin `v`.

    theta : float
        The dependence parameter (must be greater than 1).

    Returns
    -------
    u : ndarray of shape (n_observations,)
        A 1D-array of length `n_observations`, where each element is the computed
        :math:`u = h^{-1}(p \mid v)` for the corresponding pair in `X`.
    """
    X = _apply_margin_swap(X, first_margin=first_margin)

    p, v = X.T

    y = np.power(1 - v, theta)

    # No known closed-form solution, hence we use Newton method
    # with an early-stopping criterion

    # Initial guess
    x = np.power(
        (1 - v) * (np.power(1.0 - p, 1.0 / theta - 1) - 1.0) / y + 1.0,
        theta / (1.0 - theta),
    )

    max_iters = 50
    tol = 1e-8
    for _ in range(max_iters):
        k = (x - 1.0) * y
        w = np.power((1.0 / y - 1.0) * x + 1.0, 1.0 / theta)
        x_new = (
            x
            - (theta * (k - x) * (p * (-k + x) + k * w))
            / ((y - 1.0) * k - theta * y)
            / w
        )
        x_new = np.clip(x_new, 0.0, 1.0)
        diff = np.max(np.abs(x_new - x))
        x = x_new
        if diff < tol:
            break

    u = 1.0 - np.power(x, 1.0 / theta)
    return u
