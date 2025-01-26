"""Bivariate Gaussian Copula Estimation"""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st
import sklearn.utils.validation as skv

from skfolio.distribution._copula._base import _RHO_BOUNDS, BaseBivariateCopula
from skfolio.distribution._copula._utils import _apply_margin_swap


class GaussianCopula(BaseBivariateCopula):
    r"""Bivariate Gaussian Copula Estimation.

    The bivariate Gaussian copula is defined as:

    .. math::
        C_{\rho}(u, v) = \Phi_2\left(\Phi^{-1}(u), \Phi^{-1}(v) ; \rho\right)

    where:
    - :math:`\Phi_2` is the bivariate normal CDF with correlation :math:`\rho`.
    - :math:`\Phi` is the standard normal CDF and :math:`\Phi^{-1}` is its quantile function.
    - :math:`\rho \in (-1, 1)` is the correlation coefficient.

    .. note::

        Rotations are not needed for elliptical copula (e.g., Gaussian or Student-t)
        because its correlation parameter :math:`\rho \in (-1, 1)` naturally covers
        both positive and negative dependence, and they exhibit symmetric tail behavior.

    Parameters
    ----------
    use_kendall_tau_inversion : bool, default=False
        If True, :math:`\rho` is estimated using the Kendall's tau inversion method;
        otherwise, we use the MLE (Maximum Likelihood Estimation) method (default).
        The MLE is slower but more accurate.

    kendall_tau : float, optional
        If `use_kendall_tau_inversion` is True and `kendall_tau` is provided, this
        value is used; otherwise, it is computed.

    Attributes
    ----------
    rho_ : float
        Fitted parameter (:math:`\rho`) in [-1, 1].
    """

    rho_: float
    _n_params = 1

    def __init__(
        self, use_kendall_tau_inversion: bool = False, kendall_tau: float | None = None
    ):
        self.use_kendall_tau_inversion = use_kendall_tau_inversion
        self.kendall_tau = kendall_tau

    def fit(self, X: npt.ArrayLike, y=None) -> "GaussianCopula":
        r"""Fit the Bivariate Gaussian Copula.

        If `use_kendall_tau_inversion` is True, estimates :math:`\rho` using
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
        self : GaussianCopula
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        if self.use_kendall_tau_inversion:
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
                _neg_log_likelihood, args=(X,), bounds=_RHO_BOUNDS, method="bounded"
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
        cdf : ndarray of shape (n_observations, )
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

        where :math:\Phi is the standard normal CDF and :math:\Phi^{-1} is its inverse
        (the quantile function).

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        first_margin : bool, default False
            If True, compute the partial derivative with respect to the first
            margin `u`; ,otherwise, compute the partial derivative with respect to the
            second margin `v`.

        Returns
        -------
        p : ndarray of shape (n_observations, )
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

        first_margin : bool, default False
            If True, compute the inverse partial derivative with respect to the first
            margin `u`; ,otherwise, compute the inverse partial derivative with respect
            to the second margin `v`.

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
