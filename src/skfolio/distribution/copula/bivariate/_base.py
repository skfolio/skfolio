"""
Base Bivariate Copula Estimator
-------------------------------
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


class CopulaRotation(Enum):
    """
    Enum representing the rotation (in degrees) to apply to a bivariate copula.

    Attributes
    ----------
    R0 : int
        No rotation (0 degrees).
    R90 : int
        90 degrees counter-clockwise rotation.
    R180 : int
        180 degrees rotation (equivalent to a survival copula).
    R270 : int
        270 degrees counter-clockwise rotation.
    """

    R0 = 0
    R90 = 90
    R180 = 180
    R270 = 270


class BaseBivariateCopula(BaseEstimator, ABC):
    """Base Bivariate Copula Estimator


    Parameters
    ----------
    rotation : CopulaRotation, optional
       The rotation to apply to the copula (default is no rotation).
    """

    params_: dict[str, float]

    @abstractmethod
    def __init__(self, rotation: CopulaRotation = CopulaRotation.R0):
        self.rotation = rotation

    def _validate_and_rotate(self, X: npt.ArrayLike, reset: bool) -> np.ndarray:
        """
        Validate the input data and apply rotation if necessary.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input data where each row represents a bivariate observation.
            The data should be transformed to uniform marginals in [0, 1].

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        Returns
        -------
        np.ndarray
            The validated and rotated data array.
        """
        X = skv.validate_data(self, X, dtype=np.float64, reset=reset)
        if X.shape[1] != 2:
            raise ValueError("X should contains two columns for Bivariate Copula")
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError(
                "X must be uniform distributions obtained from marginals CDF "
                "transformation"
            )

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
                raise ValueError(f"Copula Rotation {self.rotation} not implemented")

        return X

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        pass

    @abstractmethod
    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        pass

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        check_is_fitted(self)

        rng = check_random_state(random_state)
        return self._scipy_model.rvs(size=n_samples, random_state=rng, **self.params_)

    # Todo avoid multiple validation
    def bic(self, X) -> float:
        log_likelihood = self.score(X)
        n_observations = X.shape[0]
        k = len(self.params_)
        bic = k * np.log(n_observations) - 2 * log_likelihood
        return bic
