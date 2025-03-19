"""Base Multivariate Distribution Estimator."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import sklearn.utils as sku

from skfolio.distribution._base import BaseDistribution


class BaseMultivariateDist(BaseDistribution, ABC):
    """Base class for Multivariate Distribution Estimators.

    This abstract class defines the interface for multivariate distribution models.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.
    """

    # Used for AIC and BIC
    _n_params: int

    def __init__(self, random_state: int | None = None):
        super().__init__(random_state=random_state)

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of model parameters."""
        pass

    @property
    @abstractmethod
    def fitted_repr(self) -> str:
        """String representation of the fitted copula."""
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseMultivariateDist":
        """Fit the multivariate distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : BaseMultivariateDist
            Returns the instance itself.
        """
        pass

    @abstractmethod
    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample (log-pdf) under the distribution
        model.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        Returns
        -------
        density : ndarray of shape (n_observations,)
            The log-likelihood of each sample under the fitted distribution model.
        """
        pass

    @abstractmethod
    def sample(
        self,
        n_samples: int = 1,
        conditioning: dict[int | str : float | tuple[float, float] | npt.ArrayLike]
        | None = None,
    ) -> np.ndarray:
        """Generate random samples from the distribution model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        conditioning : dict[int | str, float | tuple[float, float] | array-like], optional
            A dictionary specifying conditioning information for one or more assets.
            The dictionary keys are asset indices or names, and the values define how
            the samples are conditioned for that asset. Three types of conditioning
            values are supported:

            1. **Fixed value (float):**
               If a float is provided, all samples are generated under the condition
               that the asset takes exactly that value.

            2. **Bounds (tuple of two floats):**
               If a tuple `(min_value, max_value)` is provided, samples are generated
               under the condition that the asset's value falls within the specified
               bounds. Use `-np.Inf` for no lower bound or `np.Inf` for no upper bound.

            3. **Array-like (1D array):**
               If an array-like of length `n_samples` is provided, each sample is
               conditioned on the corresponding value in the array for that asset.

        Returns
        -------
        X : array-like of shape (n_samples, n_assets)
            A two-dimensional array where each row is a multivariate observation sampled
            from the fitted distribution model.
        """
        pass

    def plot_scatter_matrix(
        self,
        X: npt.ArrayLike | None = None,
        conditioning: dict[int | str : float | tuple[float, float] | npt.ArrayLike]
        | None = None,
        n_samples: int = 1000,
        title: str = "Scatter Matrix",
    ) -> go.Figure:
        """
        Plot the vine copula scatter matrix by generating samples from the fitted
        distribution model and comparing it versus the empirical distribution of `X` if
        provided.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_assets), optional
            If provided, it is used to plot the empirical scatter matrix for
            comparison versus the vine copula scatter matrix.

        conditioning : dict[int | str, float | tuple[float, float] | array-like], optional
            A dictionary specifying conditioning information for one or more assets.
            The dictionary keys are asset indices or names, and the values define how
            the samples are conditioned for that asset. Three types of conditioning
            values are supported:

            1. **Fixed value (float):**
               If a float is provided, all samples are generated under the condition
               that the asset takes exactly that value.

            2. **Bounds (tuple of two floats):**
               If a tuple `(min_value, max_value)` is provided, samples are generated
               under the condition that the asset's value falls within the specified
               bounds. Use `-np.Inf` for no lower bound or `np.Inf` for no upper bound.

            3. **Array-like (1D array):**
               If an array-like of length `n_samples` is provided, each sample is
               conditioned on the corresponding value in the array for that asset.

        n_samples : int, default=1000
            Number of samples used to control the density and readability of the plot.
            If `X` is provided and contains more than `n_samples` rows, a random
            subsample of size `n_samples` is selected. Conversely, if `X` has fewer
            rows than `n_samples`, the value is adjusted to match the number of rows in
            `X` to ensure balanced visualization.

        title : str, default="Scatter Matrix"
            The title for the plot.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A figure object containing the scatter matrix.
        """
        traces = []
        n_assets = self.n_features_in_
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X should be an 2D array")
            if X.shape[1] != n_assets:
                raise ValueError(f"X should have {n_assets} columns")
            if X.shape[0] > n_samples:
                # We subsample for improved graph readability
                rng = sku.check_random_state(self.random_state)
                indices = rng.choice(
                    np.arange(X.shape[0]), size=n_samples, replace=False
                )
                X = X[indices, :]
            else:
                # We want same proportion as X to have a balanced graph
                n_samples = X.shape[0]
            traces.append(
                go.Splom(
                    dimensions=[
                        {"label": self.feature_names_in_[i], "values": X[:, i]}
                        for i in range(n_assets)
                    ],
                    showupperhalf=False,
                    diagonal_visible=False,
                    marker=dict(
                        size=5,
                        color="rgb(85,168,104)",
                        line=dict(width=0.2, color="white"),
                        opacity=0.6,
                    ),
                    name="Historical",
                    showlegend=True,
                )
            )

        sample = self.sample(n_samples=n_samples, conditioning=conditioning)

        traces.append(
            go.Splom(
                dimensions=[
                    {"label": self.feature_names_in_[i], "values": sample[:, i]}
                    for i in range(n_assets)
                ],
                showupperhalf=False,
                diagonal_visible=False,
                marker=dict(
                    size=5,
                    color="rgb(221,132,82)",
                    line=dict(width=0.2, color="white"),
                    opacity=0.6,
                ),
                name="Generated",
                showlegend=True,
            )
        )

        if conditioning is not None:
            # Improve readability
            traces = traces[::-1]

        fig = go.Figure(data=traces)
        fig.update_layout(title=title)
        return fig
