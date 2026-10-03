"""Base lifecycle for sequential portfolio optimizers."""

# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from sklearn.utils.validation import validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization
from skfolio.typing import ArrayLike, FloatArray


class BaseOnlineOptimization(BaseOptimization):
    """Reset on fit; continue causally on partial_fit; inherit predict.

    ``weights_`` is the target for the next return period. ``previous_weights`` is an
    execution input passed to Portfolio, never a replacement for learning state.

    Parameters
    ----------
    initial_weights : array-like of shape (n_assets,), optional
        Strictly positive starting allocation summing to one. Defaults to equal
        weights. Concrete estimators may project it into their feasible set.
    previous_weights : float, dict or array-like, optional
        Actual holdings passed to portfolio evaluation, separate from learning.
    portfolio_params : dict, optional
        Parameters forwarded to the predicted portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Target allocation after the last observation, for the next return period.
    initial_weights_ : ndarray of shape (n_assets,)
        Feasible allocation before any observation.
    n_observations_ : int
        Number of successfully consumed observations, including warmup.
    """

    _requires_single_period_evaluation = True

    def __init__(
        self,
        initial_weights: ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        portfolio_params: dict | None = None,
    ) -> None:
        """Initialize allocation and portfolio evaluation parameters."""
        super().__init__(
            previous_weights=previous_weights, portfolio_params=portfolio_params
        )
        self.initial_weights = initial_weights

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseOnlineOptimization:
        """Reset learning and process returns in observation order.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Finite simple returns strictly greater than -1.
        y : Ignored
            Present for estimator compatibility.

        Returns
        -------
        self : BaseOnlineOptimization
            Fitted estimator.
        """
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> BaseOnlineOptimization:
        """Continue learning by consuming each row exactly once.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Finite simple returns strictly greater than -1. Pass a single
            observation as a two-dimensional array of shape (1, n_assets).
        y : Ignored
            Present for estimator compatibility.

        Returns
        -------
        self : BaseOnlineOptimization
            Updated estimator.

        Notes
        -----
        A failed numerical update leaves earlier successful rows committed. Input
        validation happens before processing the block. No fallback is performed.
        """
        self._validate_params()
        first_call = not hasattr(self, "n_observations_")
        X = validate_data(self, X, reset=first_call, dtype=float)
        if np.any(X <= -1):
            raise ValueError("Online optimization requires simple returns > -1.")
        if first_call:
            self._initialize(X.shape[1])
        else:
            self._validate_stream_params()
        self.error_ = None
        self.fallback_ = None
        self.fallback_chain_ = None
        for returns_t in X:
            # Compute a candidate before committing any learning state.
            weights = self._solve_update(returns_t)
            if not np.isfinite(weights).all():
                raise ValueError("The update produced non-finite weights.")
            self.weights_ = weights
            self.n_observations_ += 1
        return self

    def _validate_params(self) -> None:
        """Hook for concrete algorithm parameter checks."""

    def _validate_stream_params(self) -> None:
        """Hook for checking parameters that must remain fixed during a stream."""

    def _initialize(self, n_assets: int) -> None:
        """Validate the initial allocation and initialize learning state."""
        weights = (
            np.full(n_assets, 1 / n_assets)
            if self.initial_weights is None
            else np.array(self.initial_weights, dtype=float, copy=True)
        )
        if (
            weights.shape != (n_assets,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("initial_weights must be positive and sum to one.")
        weights /= weights.sum()
        self.initial_weights_ = weights.copy()
        self.weights_ = weights.copy()
        self.n_observations_ = 0

    def _reset(self) -> None:
        """Discard learned attributes while preserving constructor parameters."""
        for name in self._state_attributes:
            if hasattr(self, name):
                delattr(self, name)

    _state_attributes = (
        "weights_",
        "initial_weights_",
        "n_observations_",
        "n_features_in_",
        "feature_names_in_",
        "error_",
        "fallback_",
        "fallback_chain_",
    )

    @abstractmethod
    def _solve_update(self, returns_t: FloatArray) -> FloatArray:
        """Return the target for t+1 without mutating learning state."""
