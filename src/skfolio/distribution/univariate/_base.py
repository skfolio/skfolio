"""Base Univariate Estimator."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.stats as st
import sklearn.utils as sku
import sklearn.utils.validation as skv

from skfolio.distribution._base import BaseDistribution


class BaseUnivariateDist(BaseDistribution, ABC):
    """Base Univariate Distribution Estimator.

    This abstract class serves as a foundation for univariate distribution models
    based on scipy.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.
    """

    _scipy_model: st.rv_continuous

    def __init__(self, random_state: int | None = None):
        super().__init__(random_state=random_state)

    @property
    @abstractmethod
    def _scipy_params(self) -> dict[str, float]:
        """Dictionary of parameters to pass to the underlying SciPy distribution."""
        pass

    @property
    def n_params(self) -> int:
        """Number of model parameters."""
        return len(self._scipy_params)

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted univariate distribution."""
        skv.check_is_fitted(self)
        params = ", ".join([f"{k}={v:0.2g}" for k, v in self._scipy_params.items()])
        return f"{self.__class__.__name__}({params})"

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseUnivariateDist":
        """Fit the univariate distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.


        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : BaseUnivariateDist
            Returns the instance itself.
        """
        pass

    def _validate_X(self, X: npt.ArrayLike, reset: bool) -> np.ndarray:
        """Validate and convert the input data X.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        Returns
        -------
        validated_X : ndarray of shape (n_observations, 1).
            The validated input array
        """
        X = skv.validate_data(self, X, dtype=np.float64, reset=reset)
        if X.shape[1] != 1:
            raise ValueError(
                "X should contain a single column for Univariate Distribution"
            )
        return X

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            An array of points at which to evaluate the log-probability density.
            The data should be a single feature column.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            Log-likelihood values for each observation in X.
        """
        X = self._validate_X(X, reset=False)
        log_density = self._scipy_model.logpdf(X, **self._scipy_params).ravel()
        return log_density

    def sample(self, n_samples: int = 1):
        """Generate random samples from the fitted distribution.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array-like of shape (n_samples, 1)
            List of samples.
        """
        skv.check_is_fitted(self)
        rng = sku.check_random_state(self.random_state)
        sample = self._scipy_model.rvs(
            size=(n_samples, 1), random_state=rng, **self._scipy_params
        )
        return sample

    def cdf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the cumulative distribution function (CDF) for the given data.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            Data points at which to evaluate the CDF.

        Returns
        -------
        cdf : ndarray of shape (n_observations, 1)
            The CDF evaluated at each data point.
        """
        skv.check_is_fitted(self)
        return self._scipy_model.cdf(X, **self._scipy_params)

    def ppf(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the percent point function (inverse of the CDF) for the given
         probabilities.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            Probabilities for which to compute the corresponding quantiles.

        Returns
        -------
         ppf : ndarray of shape (n_observations, 1)
            The quantiles corresponding to the given probabilities.
        """
        skv.check_is_fitted(self)
        return self._scipy_model.ppf(X, **self._scipy_params)

    def plot_pdf(
        self, X: npt.ArrayLike | None = None, title: str | None = None
    ) -> go.Figure:
        """Plot the probability density function (PDF).

        Parameters
        ----------
        X : array-like of shape (n_samples, 1), optional
            If provided, it is used to plot the empirical data KDE for comparison
            versus the model PDF.

        title : str, optional
           The title for the plot. If not provided, a default title based on the fitted
           model's representation is used.

        Returns
        -------
        fig : go.Figure
           A Plotly figure object containing the PDF plot.
        """
        skv.check_is_fitted(self)
        if title is None:
            title = f"PDF of {self.__class__.__name__}"
            if X is not None:
                title += " vs Empirical KDE"

        # Compute the quantile-based range
        lower_bound = self.ppf(1e-4)
        upper_bound = self.ppf(1 - 1e-4)
        # Generate x values across this range
        x = np.linspace(lower_bound, upper_bound, 1000)

        traces = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pdfs = np.exp(self.score_samples(x.reshape(-1, 1)))
        traces.append(
            go.Scatter(
                x=x,
                y=pdfs.flatten(),
                mode="lines",
                name=self.__class__.__name__,
                line=dict(color="rgb(31, 119, 180)", dash="solid", width=1),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.17)",
                opacity=1.0,
            )
        )

        if X is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="^X has feature names", category=UserWarning
                )
                X = self._validate_X(X, reset=False)
            kde = st.gaussian_kde(X[:, 0])
            y_kde = kde(x)
            traces.append(
                go.Scatter(
                    x=x,
                    y=y_kde,
                    mode="lines",
                    name="Empirical KDE",
                    line=dict(color="rgb(85, 168, 104)", dash="dash", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(85, 168, 104, 0.17)",
                    opacity=1.0,
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="Probability Density",
        )
        fig.update_xaxes(
            tickformat=".0%",
        )
        return fig

    def qq_plot(self, X: npt.ArrayLike, title: str | None = None) -> go.Figure:
        """Plot the empirical quantiles of the sample X versus the quantiles of the
        fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1), optional
            Used to plot the empirical quantiles for comparison versus the model
            quantiles.

        title : str, optional
           The title for the plot. If not provided, a default title based on the fitted
           model's representation is used.

        Returns
        -------
        fig : go.Figure
           A Plotly figure object containing the PDF plot.
        """
        skv.check_is_fitted(self)
        if title is None:
            title = f"Q-Q Plot of {self.__class__.__name__} vs Sample Data"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="^X has feature names", category=UserWarning
            )
            X = self._validate_X(X, reset=False)

        X_sorted = np.sort(X[:, 0])
        n = len(X)

        # Compute theoretical quantiles from the model
        theoretical_quantiles = self.ppf((np.arange(1, n + 1) - 0.5) / n)

        # Create the Q-Q plot using Plotly
        fig = go.Figure(
            go.Scatter(
                x=theoretical_quantiles,
                y=X_sorted,
                mode="markers",
            )
        )
        # Add a reference line (45° line)
        min_val = min(float(theoretical_quantiles[0]), float(X_sorted[0]))
        max_val = max(float(theoretical_quantiles[-1]), float(X_sorted[-1]))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            showlegend=False,
        )
        return fig
