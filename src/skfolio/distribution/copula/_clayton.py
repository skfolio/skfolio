"""Bivariate Clayton Copula Estimation."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
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

# Clayton copula with a theta of 0.0 is just the independence copula, so we chose a
# lower bound of 1e-4. After 50, the copula is already imposing very high tail
# dependence closed to comonotonic and increasing it will make it impractical.
_THETA_BOUNDS = (1e-4, 50.0)


class ClaytonCopula(BaseBivariateCopula):
    r"""Bivariate Clayton Copula Estimation.

    The Clayton copula is an Archimedean copula characterized by strong lower tail
    dependence and little to no upper tail dependence.

    In its unrotated form, it is used for modeling extreme co-movements in the lower
    tail (i.e. simultaneous extreme losses).

    Rotations allow the copula to be adapted for different types of tail dependence:
      - A 180° rotation captures extreme co-movements in the upper tail (i.e.
        simultaneous extreme gains).

      - A 90° rotation captures scenarios where one variable exhibits extreme gains
        while the other shows extreme losses.

      - A 270° rotation captures the opposite scenario, where one variable experiences
        extreme losses while the other suffers extreme gains.

    It is defined by:

    .. math::
            C_{\theta}(u, v) = \Bigl(u^{-\theta} + v^{-\theta} - 1\Bigr)^{-1/\theta}

    where :math:`\theta > 0` is the dependence parameter. As :math:`\theta \to 0`,
    the Clayton copula converges to the independence copula. Larger values of
    :math:`\theta` result in stronger lower-tail dependence.

    .. note::

        Rotations are needed for Archimedean copulas (e.g., Joe, Gumbel, Clayton)
        because their parameters only model positive dependence, and they exhibit
        asymmetric tail behavior. To model negative dependence, one uses rotations
        to "flip" the copula's tail dependence.

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
        Fitted theta coefficient :math:`\theta` > 0.

    rotation_ : CopulaRotation
        Fitted rotation of the copula.


    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution import ClaytonCopula, compute_pseudo_observations
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
    >>> model = ClaytonCopula()
    >>>
    >>> # Fit the model to the data.
    >>> model.fit(X)
    >>>
    >>> # Display the fitted parameter and tail dependence coefficients
    >>> print(model.fitted_repr)
    ClaytonCopula(theta=0.54, rot=0°)
    >>> print(model.lower_tail_dependence)
    0.2761
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

    def fit(self, X: npt.ArrayLike, y=None) -> "ClaytonCopula":
        r"""Fit the Bivariate Clayton Copula.

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

            # For Clayton, the theoretical relationship is: tau = theta/(theta+2)
            abs_kendall_tau = min(abs(kendall_tau), 0.9999)

            self.theta_ = np.clip(
                2 * abs_kendall_tau / (1 - abs_kendall_tau),
                a_min=_THETA_BOUNDS[0],
                a_max=_THETA_BOUNDS[1],
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
        """Compute the CDF of the bivariate Clayton copula.

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
        r"""Compute the h-function (partial derivative) for the bivariate Clayton copula
        with respect to a specified margin.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math::  \begin{aligned}
                   C(u,v)&=\Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta},\\[6pt]
                   h(u \mid v)
                     &= \frac{\partial C(u,v)}{\partial v}
                     = \Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta-1}\,v^{-\theta-1}.
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
            h-function values :math:`h(u \mid v)` for each observation in X.
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
        known as the inverse h-function.

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
        """
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
        r"""Compute the log-likelihood of each sample (log-pdf) under the model.

        For Clayton, the PDF is given by:

        .. math:: c(u,v) = (\theta+1)\,\Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-\frac{1}{\theta}-2}\,(u\,v)^{-\theta-1}

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
        if self.rotation_ == CopulaRotation.R0:
            return np.power(2.0, -1.0 / self.theta_)
        return 0

    @property
    def upper_tail_dependence(self) -> float:
        """Theoretical upper tail dependence coefficient."""
        skv.check_is_fitted(self)
        if self.rotation_ == CopulaRotation.R180:
            return np.power(2.0, -1.0 / self.theta_)
        return 0

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted copula."""
        return (
            f"{self.__class__.__name__}(theta={self.theta_:0.2f}, rot={self.rotation_})"
        )


def _neg_log_likelihood(theta: float, X: np.ndarray) -> float:
    """Negative log-likelihood function for the Clayton copula.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    theta : float
        The dependence parameter (must be greater than 0).

    Returns
    -------
    value : float
        The negative log-likelihood value.
    """
    return -np.sum(_base_sample_scores(X=X, theta=theta))


def _base_sample_scores(X: np.ndarray, theta: float) -> np.ndarray:
    r"""Compute the log-likelihood of each sample (log-pdf) under the bivariate Clayton
    copula.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        Bivariate samples `(u, v)`, with each component in [0,1].

    theta : float
        The dependence parameter (must be greater than 0).

    Returns
    -------
    logpdf : ndarray of shape (n_observations,)
        Log-likelihood values for each observation.

    Raises
    ------
    ValueError
        If theta is not greater than 0.
    """
    if theta <= 0:
        raise ValueError("Theta must be greater than 1 for the Clayton copula.")

    x, y = np.log(X).T

    log_density = (
        np.log1p(theta)
        - (2.0 + 1.0 / theta) * np.log1p(np.expm1(-theta * x) + np.expm1(-theta * y))
        - (1.0 + theta) * (x + y)
    )
    return log_density


def _base_cdf(X: np.ndarray, theta: float) -> np.ndarray:
    r"""Bivariate Clayton CDF (unrotated).

    .. math::
        C(u,v) = \Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta}.
    """
    cdf = np.power(np.sum(np.power(X, -theta), axis=1) - 1, -1.0 / theta)
    return cdf


def _base_partial_derivative(
    X: np.ndarray, first_margin: bool, theta: float
) -> np.ndarray:
    r"""
    Compute the partial derivative (h-function) for the unrotated Clayton copula.

    For Clayton, the copula is defined as:

    .. math::
        C(u,v)=\Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta}.

    The partial derivative with respect to v is:

    .. math::
        \frac{\partial C(u,v)}{\partial v} = \Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta-1}\,v^{-\theta-1}.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
         An array of bivariate inputs `(u, v)` with values in [0, 1].

    first_margin : bool, default=False
         If True, compute with respect to u (by swapping margins); otherwise
         compute with respect to v.

    theta : float
         The dependence parameter (must be > 0).

    Returns
    -------
    p : ndarray of shape (n_observations,)
         The computed h-function values.
    """
    X = _apply_margin_swap(X, first_margin=first_margin)
    x = np.power(X[:, 0], -theta)
    y = np.power(X[:, 1], theta)
    p = np.power(1.0 + y * (x - 1.0), -(1.0 + 1.0 / theta))
    return p


def _base_inverse_partial_derivative(
    X: np.ndarray, first_margin: bool, theta: float
) -> np.ndarray:
    r"""
    Compute the inverse partial derivative for the unrotated Clayton copula,
    i.e. solve for u in h(u|v)=p.

    In other words, given
      - p, the value of the h-function, and
      - v, the conditioning variable,
    solve:

    .. math::
      p = \Bigl(u^{-\theta}+v^{-\theta}-1\Bigr)^{-1/\theta-1}\,v^{-\theta-1},

    for u ∈ [0,1]. Since no closed-form solution exists, we use a Newton method.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
         An array with first column p (h-function values) and second column v
         (conditioning variable).

    first_margin : bool, default=False
         If True, treat the first margin as the conditioning variable.

    theta : float
         The dependence parameter (must be > 0).

    Returns
    -------
    u : ndarray of shape (n_observations,)
         A 1D-array where each element is the solution u ∈ [0,1] such that h(u|v)=p.
    """
    X = _apply_margin_swap(X, first_margin=first_margin)
    x = np.power(X[:, 0], -theta / (theta + 1.0))
    y = np.power(X[:, 1], -theta)
    u = np.power(1.0 + y * (x - 1.0), -1.0 / theta)
    return u
