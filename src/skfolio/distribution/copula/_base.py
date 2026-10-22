"""Base Bivariate Copula Estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.utils as sku
import sklearn.utils.validation as skv

from skfolio.distribution._base import BaseDistribution
from skfolio.distribution.copula._utils import (
    empirical_tail_concentration,
    plot_tail_concentration,
)
from skfolio.typing import ArrayLike, FloatArray, StrArray

UNIFORM_MARGINAL_EPSILON = 1e-9
_RHO_BOUNDS = (-0.999, 0.999)


class BaseBivariateCopula(BaseDistribution, ABC):
    """Base class for Bivariate Copula Estimators.

    This abstract class defines the interface for bivariate copula models, including
    methods for fitting, sampling, scoring, and computing partial derivatives.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.
    """

    # Used for AIC and BIC
    _n_params: int
    feature_names_in_: StrArray

    def __init__(self, random_state: int | None = None) -> None:
        super().__init__(random_state=random_state)

    def _validate_X(
        self, X: ArrayLike, reset: bool, *, clip: bool = True
    ) -> FloatArray:
        """Validate the input data.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` must be in the interval `[0, 1]`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        clip : bool, default=True
            Whether to clip the margins to `[1e-9, 1 - 1e-9]` to avoid singular
            logarithms, inverse CDFs and divisions. Set to False for CDF and
            joint survival calculations, which preserve the original thresholds
            and handle boundaries exactly. All input checks still apply.

        Returns
        -------
        validated_X: ndarray of shape (n_observations, 2)
            The validated data array.

        Raises
        ------
        ValueError
            If input data is invalid (e.g., not in `[0, 1]` or incorrect shape).
        """
        X = skv.validate_data(self, X, dtype=np.float64, reset=reset)
        if X.shape[1] != 2:
            raise ValueError("X must contain two columns for Bivariate Copula")
        if not np.all((X >= 0) & (X <= 1)):
            raise ValueError(
                "X must be in the interval `[0, 1]`, usually reprinting uniform "
                "distributions obtained from marginals CDF transformation"
            )

        if clip:
            X = np.clip(X, UNIFORM_MARGINAL_EPSILON, 1 - UNIFORM_MARGINAL_EPSILON)
        return X

    @property
    def n_params(self) -> int:
        """Number of model parameters."""
        return self._n_params

    @property
    @abstractmethod
    def lower_tail_dependence(self) -> float:
        """Theoretical lower tail dependence coefficient."""
        ...

    @property
    @abstractmethod
    def upper_tail_dependence(self) -> float:
        """Theoretical upper tail dependence coefficient."""
        ...

    @property
    @abstractmethod
    def fitted_repr(self) -> str:
        """String representation of the fitted copula."""
        ...

    @abstractmethod
    def fit(self, X: ArrayLike, y: None = None) -> BaseBivariateCopula:
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
        ...

    @abstractmethod
    def cdf(self, X: ArrayLike) -> FloatArray:
        """Compute the CDF of the bivariate copula.

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
        ...

    def survival_probability(self, X: ArrayLike) -> FloatArray:
        r"""Return the probability that both variables exceed their respective
        thresholds.

        Parameters
        ----------
        X : array-like of shape (n_observations, 2)
            Uniform marginal probability levels `(u, v)` in `[0, 1]`. Each row
            specifies the thresholds for the two variables.

        Returns
        -------
        probabilities : ndarray of shape (n_observations,)
            Joint probabilities :math:`P(U > u, V > v)` for each row in X.

        See Also
        --------
        cdf : Probability that both variables are below their respective thresholds.
        tail_concentration : Conditional probabilities at a common quantile.

        Notes
        -----
        For uniform margins, the joint survival probability is

        .. math::
            S(u, v) = 1 - u - v + C(u, v).

        The default implementation uses the CDF identity. Its precision in very
        small tails is limited by the CDF and subtraction. Subclasses can provide
        a more accurate calculation, such as reflection for elliptical copulas.

        Examples
        --------
        >>> from skfolio.distribution import IndependentCopula
        >>> copula = IndependentCopula().fit([[0.2, 0.3], [0.7, 0.8]])
        >>> copula.survival_probability([[0.5, 0.75]])
        array([0.125])
        """
        # cdf checks fitted state and validates X, including DataFrame column names.
        cdf = self.cdf(X)
        X = np.asarray(X, dtype=np.float64)
        u, v = X.T
        probabilities = (1.0 - u) - (v - cdf)
        # Subtraction can round outside the Frechet bounds.
        lower = np.maximum((1.0 - u) - v, 0.0)
        upper = 1.0 - np.maximum(u, v)
        return np.clip(probabilities, lower, upper)

    @abstractmethod
    def partial_derivative(
        self, X: ArrayLike, first_margin: bool = False
    ) -> FloatArray:
        r"""Compute the h-function (partial derivative) for the bivariate copula
        with respect to a specified margin.

        The h-function with respect to the second margin represents the conditional
        distribution function of :math:`u` given :math:`v`:

        .. math::
            h(u \mid v) = \frac{\partial C(u,v)}{\partial v}

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
        ...

    @abstractmethod
    def inverse_partial_derivative(
        self, X: ArrayLike, first_margin: bool = False
    ) -> FloatArray:
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
        ...

    @abstractmethod
    def score_samples(self, X: ArrayLike) -> FloatArray:
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
        ...

    def sample(self, n_samples: int = 1) -> FloatArray:
        """Generate random samples from the bivariate copula using the inverse
        Rosenblatt transform.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array-like of shape (n_samples, 2)
            An array of bivariate inputs `(u, v)` where each row represents a
            bivariate observation. Both `u` and `v` are uniform marginals in the
            interval `[0, 1]`.
        """
        skv.check_is_fitted(self)
        rng = sku.check_random_state(self.random_state)

        # Generate independent Uniform(0, 1) samples
        X = rng.random(size=(n_samples, 2))

        # Apply the inverse Rosenblatt transform on the first variable.
        X[:, 1] = self.inverse_partial_derivative(X, first_margin=True)
        return X

    def tail_concentration(self, quantiles: ArrayLike) -> FloatArray:
        r"""
        Compute the tail concentration function for a set of quantiles.

        The tail concentration function is defined as:

        .. math::
            T(q) = \begin{cases}
                \frac{P(U \le q, V \le q)}{q} & \text{if } q \le 0.5 \\
                \frac{P(U > q, V > q)}{1 - q} & \text{if } q > 0.5
            \end{cases}

        where :math:`U` and :math:`V` are the pseudo-observations of the first and
        second variables, respectively, and :math:`0 < q < 1`. These are
        conditional probabilities given that the first component is in the
        corresponding tail.

        Parameters
        ----------
        quantiles : array-like of shape (n_quantiles,)
           A 1D array of quantile levels (values between 0 and 1) at which to compute
           the tail concentration.

        Returns
        -------
        concentration : ndarray of shape (n_quantiles,)
           The computed tail concentration values corresponding to each quantile.
           At zero and one, the theoretical lower and upper tail dependence
           limits are returned because the conditioning event has probability
           zero and the conditional probability is undefined. An empty input
           returns an empty array.

        Raises
        ------
        ValueError
           If `quantiles` is not one-dimensional or contains values outside [0, 1].

        See Also
        --------
        survival_probability : Joint upper-tail probabilities.
        lower_tail_dependence : Limit of the lower tail concentration at zero.
        upper_tail_dependence : Limit of the upper tail concentration at one.

        References
        ----------
        .. [1] "Quantitative Risk Management: Concepts, Techniques, and Tools",
            McNeil, Frey, Embrechts (2005)
        """
        skv.check_is_fitted(self)
        quantiles = np.asarray(quantiles, dtype=float)
        if quantiles.ndim != 1:
            raise ValueError("quantiles must be a one-dimensional array.")
        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            raise ValueError("quantiles must be between 0.0 and 1.0.")

        X = np.column_stack((quantiles, quantiles))
        if hasattr(self, "feature_names_in_"):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        concentration = np.empty_like(quantiles)
        concentration[quantiles == 0] = self.lower_tail_dependence
        concentration[quantiles == 1] = self.upper_tail_dependence
        lower = (quantiles > 0) & (quantiles <= 0.5)
        upper = (quantiles > 0.5) & (quantiles < 1)
        if np.any(lower):
            concentration[lower] = self.cdf(X[lower]) / quantiles[lower]
        if np.any(upper):
            concentration[upper] = self.survival_probability(X[upper]) / (
                1.0 - quantiles[upper]
            )
        return concentration

    def plot_tail_concentration(
        self, X: ArrayLike | None = None, title: str | None = None
    ) -> go.Figure:
        r"""
        Plot the tail concentration function.

        This method computes the tail concentration function at 100 evenly spaced
        quantile levels between 0.005 and 0.995.
        The plot displays the concentration values on the y-axis and the quantile levels
        on the x-axis.

        The lower and upper tail concentrations are defined as:

        .. math::
            \lambda_L(q) = P(U_2 \le q \mid U_1 \le q)

            \lambda_U(q) = P(U_2 \ge q \mid U_1 \ge q)

        where :math:`U_1` and :math:`U_2` are the pseudo-observations of the first and
        second variables, respectively.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2), optional
            If provided, it is used to plot the empirical tail concentration for
            comparison versus the model tail concentration.

        title : str, optional
            The title for the plot. If not provided, a default title based on the fitted
            copula's representation is used.

        Returns
        -------
        fig : go.Figure
            A Plotly figure object containing the tail concentration curve.

        References
        ----------
        .. [1] "Quantitative Risk Management: Concepts, Techniques, and Tools",
            McNeil, Frey, Embrechts (2005)
        """
        if title is None:
            title = f"Tail Concentration of Bivariate {self.__class__.__name__}"
            if X is not None:
                title += " vs Empirical"

        quantiles = np.linspace(5e-3, 1.0 - 5e-3, num=100)
        concentration = self.tail_concentration(quantiles)

        tail_concentration_dict = {self.__class__.__name__: concentration}
        if X is not None:
            tail_concentration_dict["Empirical"] = empirical_tail_concentration(
                X, quantiles=quantiles
            )

        fig = plot_tail_concentration(
            tail_concentration_dict=tail_concentration_dict,
            quantiles=quantiles,
            title=title,
            smoothing=1.3,
        )
        return fig

    def plot_pdf_2d(self, title: str | None = None) -> go.Figure:
        """
        Plot a 2D contour of the estimated probability density function (PDF).

        This method generates a grid over [0, 1]^2, computes the PDF, and displays a
        contour plot of the PDF.
        Contour levels are limited to the 97th quantile to avoid extreme densities.

        Parameters
        ----------
        title : str, optional
           The title for the plot. If not provided, a default title based on the fitted
           copula's representation is used.

        Returns
        -------
        fig : go.Figure
           A Plotly figure object containing the 2D contour plot of the PDF.
        """
        skv.check_is_fitted(self)

        if title is None:
            title = f"PDF of the Bivariate {self.__class__.__name__}"

        u = np.linspace(0.01, 0.99, 100)
        U, V = np.meshgrid(u, u)
        grid_points = np.column_stack((U.ravel(), V.ravel()))
        pdfs = np.exp(self.score_samples(grid_points)).reshape(U.shape)
        # After the 97th quantile, the pdf gets too dense, and it dilutes the plot.
        end = round(np.quantile(pdfs, 0.97), 1)
        fig = go.Figure(
            data=go.Contour(
                x=u,
                y=u,
                z=pdfs,
                colorscale="Magma",
                contours=dict(start=0, end=end, size=0.2),
                line=dict(width=0),
                colorbar=dict(title="PDF"),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="u",
            yaxis_title="v",
        )
        return fig

    def plot_pdf_3d(self, title: str | None = None) -> go.Figure:
        """
        Plot a 3D surface of the estimated probability density function (PDF).

        This method generates a grid over [0, 1]^2, computes the PDF, and displays a
        3D surface plot of the PDF using Plotly.

        Parameters
        ----------
        title : str, optional
           The title for the plot. If not provided, a default title based on the fitted
           copula's representation is used.

        Returns
        -------
        fig : go.Figure
           A Plotly figure object containing a 3D surface plot of the PDF.
        """
        skv.check_is_fitted(self)

        if title is None:
            title = f"PDF of the Bivariate {self.__class__.__name__}"

        u = np.linspace(0.03, 0.97, 100)
        U, V = np.meshgrid(u, u)
        grid_points = np.column_stack((U.ravel(), V.ravel()))
        pdfs = np.exp(self.score_samples(grid_points)).reshape(U.shape)
        fig = go.Figure(data=[go.Surface(x=U, y=V, z=pdfs, colorscale="Magma")])
        fig.update_layout(
            title=title, scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="PDF")
        )
        return fig
