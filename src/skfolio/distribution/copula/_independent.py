"""Bivariate Independent Copula Estimation."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.distribution.copula._base import BaseBivariateCopula


class IndependentCopula(BaseBivariateCopula):
    r"""Bivariate Independent Copula (also called the product copula).

    It is defined by:

    .. math::
        C(u, v) = u \cdot v

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

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

    _n_params = 0

    def __init__(self, random_state: int | None = None):
        super().__init__(random_state=random_state)

    def fit(self, X: npt.ArrayLike, y=None) -> "IndependentCopula":
        """Fit the Bivariate Independent Copula.

        Provided for compatibility with the API.

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
        self : IndependentCopula
            Returns the instance itself.
        """
        _ = self._validate_X(X, reset=True)
        return self

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the CDF of the bivariate Independent copula.

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
        cdf = X.prod(axis=1)
        return cdf

    def partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the h-function (partial derivative) for the bivariate Independent
        copula.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math::
            \frac{\partial C(u,v)}{\partial v}=u,

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Array of pairs :math:`(u,v)`, where each value is in the interval [0,1].

        Returns
        -------
        np.ndarray
            Array of h-function values for each observation in X.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        h = X[:, 1] if first_margin else X[:, 0]
        return h

    def inverse_partial_derivative(
        self, X: npt.ArrayLike, first_margin: bool = False
    ) -> np.ndarray:
        r"""Compute the inverse of the bivariate copula's partial derivative, commonly
        known as the inverse h-function.

        For the independent copula, the h-function with respect to the second margin is

        .. math::
             h(u\mid v)= u,

        and the derivative with respect to the first margin is

        .. math::
             g(u,v)= v.

        Their inverses are trivial:

          - Given (p,v) for h(u|v)= p, we have u = p.
          - Given (p,u) for g(u,v)= p, we have v = p.

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
        u = X[:, 1] if first_margin else X[:, 0]
        return u

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
         X : array-like of shape (n_samples, 2)
            The input data where each row represents a bivariate observation.
            The data should be transformed to uniform marginals in [0, 1].

        Returns
        -------
        density : ndarray of shape (n_samples,)
            The log-likelihood of each sample under the fitted copula.
        """
        skv.check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.zeros(X.shape[0])  # log(1.0)

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
        return f"{self.__class__.__name__}()"
