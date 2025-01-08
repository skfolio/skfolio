"""
Base Univariate Estimator
-------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy.stats as st
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv


class BaseUnivariate(skb.BaseEstimator):
    """Base Univariate Distribution Estimator.

    Parameters
    ----------

    """

    params_: dict[str, float]
    _scipy_model: st.rv_continuous

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
        skv.check_is_fitted(self)
        X = skv.validate_data(self, X, dtype=np.float64, reset=False)
        if X.shape[1] != 1:
            raise ValueError(
                "X should should contain a single column for Univariate Distribution"
            )

        log_density = self._scipy_model.logpdf(X, **self.params_)
        return log_density

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
        skv.check_is_fitted(self)

        rng = sku.check_random_state(random_state)
        return self._scipy_model.rvs(size=n_samples, random_state=rng, **self.params_)

    # Todo avoid multiple validation
    def bic(self, X) -> float:
        log_likelihood = self.score(X)
        n_observations = X.shape[0]
        k = len(self.params_)
        bic = k * np.log(n_observations) - 2 * log_likelihood
        return bic

    def cdf(self, X) -> np.ndarray:
        skv.check_is_fitted(self)
        return self._scipy_model.cdf(X, **self.params_)
