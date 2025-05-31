"""Bivariate Gaussian Copula Estimation."""

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

from skfolio.distribution.copula._base import _RHO_BOUNDS, BaseBivariateCopula
from skfolio.distribution.copula._utils import _apply_margin_swap


class GaussianCopula(BaseBivariateCopula):
    r"""Bivariate Gaussian Copula Estimation.

    The bivariate Gaussian copula is defined as:

    .. math::
        C_{\rho}(u, v) = \Phi_2\left(\Phi^{-1}(u), \Phi^{-1}(v) ; \rho\right)

    where:
        - :math:`\Phi_2` is the bivariate normal CDF with correlation :math:`\rho`.
        - :math:`\Phi` is the standard normal CDF and :math:`\Phi^{-1}` its quantile function.
        - :math:`\rho \in (-1, 1)` is the correlation coefficient.

    .. note::

        Rotations are not needed for elliptical copula (e.g., Gaussian or Student-t)
        because its correlation parameter :math:`\rho \in (-1, 1)` naturally covers
        both positive and negative dependence, and they exhibit symmetric tail behavior.

    Parameters
    ----------
    itau : bool, default=True
        If True, :math:`\rho` is estimated using the Kendall's tau inversion method;
        otherwise, we use the MLE (Maximum Likelihood Estimation) method. The MLE is
        slower but more accurate.

    kendall_tau : float, optional
        If `itau` is True and `kendall_tau` is provided, this
        value is used; otherwise, it is computed.

    tolerance : float, default=1e-4
        Convergence tolerance for the MLE optimization.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Attributes
    ----------
    rho_ : float
        Fitted parameter (:math:`\rho`) in [-1, 1].

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution import GaussianCopula, compute_pseudo_observations
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
    >>> model = GaussianCopula()
    >>>
    >>> # Fit the model to the data.
    >>> model.fit(X)
    >>>
    >>> # Display the fitted parameter and tail dependence coefficients
    >>> print(model.fitted_repr)
    GaussianCopula(rho=0.327)
    >>> print(model.lower_tail_dependence)
    0.0
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

    rho_: float
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

    def fit(self, X: npt.ArrayLike, y=None) -> "GaussianCopula":
        r"""Fit the Bivariate Gaussian Copula.

        If `itau` is True, estimates :math:`\rho` using Kendall's tau inversion.
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
        self : GaussianCopula
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        if self.itau:
            if self.kendall_tau is None:
                kendall_tau = st.kendalltau(X[:, 0], X[:, 1]).statistic
            else:
                kendall_tau = self.kendall_tau
            self.rho_ = np.clip(
                np.sin((np.pi * kendall_tau) / 2.0),
                a_min=_RHO_BOUNDS[0],
                a_max=_RHO_BOUNDS[1],
            )

        else:
            result = so.minimize_scalar(
                _neg_log_likelihood,
                args=(X,),
                bounds=_RHO_BOUNDS,
                method="bounded",
                options={"xatol": self.tolerance},
            )
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            self.rho_ = result.x

        return self

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the CDF of the bivariate Gaussian copula.

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
        cdf = st.multivariate_normal.cdf(
            x=sp.ndtri(X),
            mean=np.array([0, 0]),
            cov=np.array([[1, self.rho_], [self.rho_, 1]]),
        )
        return cdf

    def partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the h-function (partial derivative) for the bivariate Gaussian
        copula.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math:: \begin{aligned}
                  h(u \mid v) &= \frac{\partial C(u,v)}{\partial v} \\
                  &= \Phi\Bigl(\frac{\Phi^{-1}(u)-\rho\,\Phi^{-1}(v)}{\sqrt{1-\rho^2}}\Bigr)
                  \end{aligned}

        where :math:`\Phi` is the standard normal CDF and :math:`\Phi^{-1}` is its
        inverse (the quantile function).

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
        p : ndarray of shape (n_observations,)
            h-function values :math:`h(u \mid v) \;=\; p` for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        X = _apply_margin_swap(X, first_margin=first_margin)
        # Compute the inverse CDF (percent point function) using ndtri for better
        # performance
        u_inv, v_inv = sp.ndtri(X).T
        p = sp.ndtr((u_inv - self.rho_ * v_inv) / np.sqrt(1 - self.rho_**2))
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
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        X = _apply_margin_swap(X, first_margin=first_margin)
        p_inv, v_inv = sp.ndtri(X).T
        u_inv = p_inv * np.sqrt(1 - self.rho_**2) + self.rho_ * v_inv
        u = sp.ndtr(u_inv)
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
        log_density = _base_sample_scores(X=X, rho=self.rho_)
        return log_density

    @property
    def lower_tail_dependence(self) -> float:
        """Theoretical lower tail dependence coefficient."""
        return 0

    @property
    def upper_tail_dependence(self) -> float:
        """Theoretical upper tail dependence coefficient."""
        return 0

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted copula."""
        return f"{self.__class__.__name__}(rho={self.rho_:0.3f})"


def _neg_log_likelihood(rho: float, X: np.ndarray) -> float:
    """Negative log-likelihood function for optimization.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    rho : float
        Correlation copula parameter.

    Returns
    -------
    value : float
        The negative log-likelihood value.
    """
    return -np.sum(_base_sample_scores(X=X, rho=rho))


def _base_sample_scores(X: np.ndarray, rho: float) -> np.ndarray:
    """Compute the log-likelihood of each sample (log-pdf) under the bivariate
    Gaussian copula model.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
        having been transformed to uniform marginals.

    rho : float
        Gaussian copula parameter.

    Returns
    -------
    density : ndarray of shape (n_observations,)
        The log-likelihood of each sample under the fitted copula.

    Raises
    ------
    ValueError
        If rho is not in (-1, 1)
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1 and 1.")

    # Inverse CDF (ppf) using stdtrit for better performance
    u_inv, v_inv = sp.ndtri(X).T

    # Using np.log1p to avoid loss of precision
    log_density = -0.5 * np.log1p(-(rho**2)) - rho * (
        0.5 * rho * (u_inv**2 + v_inv**2) - u_inv * v_inv
    ) / (1 - rho**2)
    return log_density
