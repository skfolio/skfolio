"""Base Distribution Estimator."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from enum import auto

import numpy as np
import numpy.typing as npt
import sklearn.base as skb

from skfolio.utils.tools import AutoEnum


class SelectionCriterion(AutoEnum):
    """Enum representing the selection criteria.

    Attributes
    ----------
    AIC : str
        Akaike Information Criterion (AIC)

    BIC : str
        Bayesian Information Criterion (BIC)
    """

    AIC = auto()
    BIC = auto()


class BaseDistribution(skb.BaseEstimator, ABC):
    """Base Distribution Estimator.

    This abstract class serves as a foundation for distribution models in skfolio.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of model parameters."""
        pass

    @property
    @abstractmethod
    def fitted_repr(self) -> str:
        """String representation of the fitted model."""
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseDistribution":
        """Fit the univariate distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            The input data.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : BaseDistribution
            Returns the instance itself.
        """
        pass

    @abstractmethod
    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            The input data.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            Log-likelihood values for each observation in X.
        """
        pass

    def sample(self, n_samples: int = 1):
        """Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array-like of shape (n_samples, 1)
            List of samples.
        """
        pass

    def score(self, X: npt.ArrayLike, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            An array of data points for which the total log-likelihood is computed.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        logprob : float
            The total log-likelihood (sum of log-pdf values).
        """
        return np.sum(self.score_samples(X))

    def aic(self, X: npt.ArrayLike) -> float:
        r"""Compute the Akaike Information Criterion (AIC) for the model given data X.

        The AIC is defined as:

        .. math::
            \mathrm{AIC} = -2 \, \log L \;+\; 2 k,

        where

        - :math:`\log L` is the total log-likelihood
        - :math:`k` is the number of parameters in the model

        A lower AIC value indicates a better trade-off between model fit and complexity.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            The input data on which to compute the AIC.

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
        log_likelihood = self.score(X)
        return 2 * (self.n_params - log_likelihood)

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
        X : array-like of shape (n_observations, n_features)
            The input data on which to compute the BIC.

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
        return -2 * log_likelihood + self.n_params * np.log(n)
