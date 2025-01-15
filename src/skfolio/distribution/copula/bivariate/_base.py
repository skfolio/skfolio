"""
Base Bivariate Copula Estimator
-------------------------------
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv

_UNIFORM_MARGINAL_EPSILON = 1e-8
_RHO_BOUNDS = (-0.999, 0.999)


class CopulaRotation(Enum):
    """Enum representing the rotation (in degrees) to apply to a bivariate copula.

    Attributes
    ----------
    R0 : int
        No rotation (0°).
    R90 : int
        90° rotation.
    R180 : int
        180° rotation.
    R270 : int
        270° rotation.
    """

    R0 = 0
    R90 = 90
    R180 = 180
    R270 = 270


class BaseBivariateCopula(skb.BaseEstimator, ABC):
    """Base class for Bivariate Copula Estimators.

    This abstract class defines the interface for bivariate copula models, including
    methods for fitting, sampling, scoring, and computing partial derivatives.
    """

    # Used for AIC and BIC
    _n_params: int

    def _validate_X(self, X: npt.ArrayLike, reset: bool) -> np.ndarray:
        """Validate the input data and apply rotation if necessary.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        Returns
        -------
        validated_X: ndarray of shape (n_observations, 2)
            The validated and rotated data array.

        Raises
        ------
        ValueError
            If input data is invalid (e.g., not in `[0, 1]` or incorrect shape).
        """
        X = skv.validate_data(self, X, dtype=np.float64, reset=reset)
        if X.shape[1] != 2:
            raise ValueError("X must contains two columns for Bivariate Copula")
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError(
                "X must be in the interval `[0, 1]`, usually reprinting uniform "
                "distributions obtained from marginals CDF transformation"
            )

        # Handle potential numerical issues by ensuring X doesn't contain exact 0 or 1.
        X = np.clip(X, _UNIFORM_MARGINAL_EPSILON, 1 - _UNIFORM_MARGINAL_EPSILON)

        if hasattr(self, "rotation"):
            match self.rotation:
                case CopulaRotation.R0:
                    pass
                case CopulaRotation.R90:
                    X[:, 0] = 1 - X[:, 0]
                case CopulaRotation.R180:
                    X = 1 - X
                case CopulaRotation.R270:
                    X[:, 1] = 1 - X[:, 1]
                case _:
                    raise ValueError(f"Unsupported rotation: {self.rotation}")
        return X

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None):
        """Fit the copula model.

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
        self : BaseBivariateCopula
            Returns the instance itself.
        """
        pass

    @abstractmethod
    def partial_derivative(self, X: npt.ArrayLike) -> np.ndarray:
        r"""
        Compute the bivariate copula's partial derivative (h-function).
        For a bivariate copula :math:`C(u,v)`, the h-function is defined as

        .. math::
            h(u \mid v) = \frac{\partial C(u,v)}{\partial v}

        which represents the conditional distribution function of :math:`u` given
        :math:`v`.

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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        value : float
            Total log-likelihood of the input data.
        """
        return np.sum(self.score_samples(X))

    def aic(self, X: npt.ArrayLike) -> float:
        r"""Compute the Akaike Information Criterion (AIC) for the model given data X.

        The AIC is defined as:

        .. math::
            \mathrm{AIC} = -2 \, \log L \;+\; 2 k,

        where

        - :math:`\log L` is the (maximized) total log-likelihood
        - :math:`k` is the number of parameters in the model

        A lower AIC value indicates a better trade-off between model fit and complexity.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Notes
        -----
        In practice, both AIC and BIC measure the trade-off between model fit and
        complexity, but BIC tends to prefer simpler models for large :math:`n`
        because of the :math:`\ln(n)` term.

        Returns
        -------
        aic : float
            The AIC of the fitted model on the given data.

        References
        ----------
        .. [1] "A new look at the statistical model identification", Akaike (1974).
        """
        # Example implementation pattern (pseudo-code):
        log_likelihood = self.score(X)
        return 2 * (self._n_params - log_likelihood)

    def bic(self, X: npt.ArrayLike) -> float:
        r"""Compute the Bayesian Information Criterion (BIC) for the model given data X.

        The BIC is defined as:

        .. math::
            \mathrm{BIC} = -2 \, \log L \;+\; k \,\ln(n),

        where

        - :math:`\log L` is the (maximized) total log-likelihood
        - :math:`k` is the number of parameters in the model
        - :math:`n` is the number of observations

        A lower BIC value suggests a better fit while imposing a stronger penalty
        for model complexity than the AIC.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`,
            having been transformed to uniform marginals.

        Returns
        -------
        bic : float
            The BIC of the fitted model on the given data.

        Notes
        -----
        In practice, both AIC and BIC measure the trade-off between model fit and
        complexity, but BIC tends to prefer simpler models for large :math:`n`
        because of the :math:`\ln(n)` term.

        References
        ----------
        .. [1]  "Estimating the dimension of a model", Schwarz, G. (1978).
        """
        log_likelihood = self.score(X)
        n = X.shape[0]
        k = self._n_params
        return -2 * log_likelihood + k * np.log(n)

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the bivariate copula using the inverse
        Rosenblatt transform.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the sample generation.

        Returns
        -------
        X : array-like of shape (n_samples, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` are uniform marginals in the
            interval `[0, 1]`.
        """
        skv.check_is_fitted(self)
        rng = sku.check_random_state(random_state)

        # Generate independent Uniform(0, 1) samples
        X = rng.random(size=(n_samples, 2))

        # Apply the inverse Rosenblatt transform on the first variable.
        X[:, 0] = self.inverse_partial_derivative(X)

        return X
