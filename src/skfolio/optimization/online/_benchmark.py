"""Benchmark estimators for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from skfolio.measures import get_cumulative_returns
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.naive import EqualWeighted


def net_to_relatives(X: npt.ArrayLike) -> np.ndarray:
    """Validate comparator returns without clipping bankrupt assets."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.all(np.isfinite(values)):
        raise ValueError("X must be a nonempty finite 2D return matrix")
    if np.any(values <= -1):
        raise ValueError("Returns must be greater than -1")
    return 1.0 + values


def _constant_wealth_history(
    relatives: np.ndarray,
    weights: np.ndarray | None,
    initial_wealth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gross hindsight wealth and fixed targets with zero-return cash."""
    if weights is None:
        raise ValueError("Cannot compute hindsight wealth without fitted weights")
    if (
        np.ndim(initial_wealth) != 0
        or not np.isfinite(initial_wealth)
        or initial_wealth <= 0
    ):
        raise ValueError("initial_wealth must be a finite positive scalar")
    portfolio_returns = (relatives - 1.0) @ weights
    wealth = np.concatenate(
        (
            [initial_wealth],
            get_cumulative_returns(
                returns=portfolio_returns, compounded=True, base=initial_wealth
            ),
        )
    )
    targets = np.broadcast_to(weights, relatives.shape).copy()
    return wealth, targets


class ConstantRebalancedPortfolio(BaseOptimization):
    """Portfolio rebalanced to fixed weights each period.

    Keeps a fixed portfolio w and rebalances to it every period.

    Parameters
    ----------
    weights : array-like of shape (n_assets,), optional
        Fixed weights to rebalance to. If None, uses
        :class:`~skfolio.optimization.EqualWeighted` to obtain uniform weights.
    portfolio_params : dict, optional
        Portfolio parameters propagated to the produced portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Portfolio weights.

    wealth_ : float
        Gross hindsight wealth after all periods, excluding costs and fees.

    all_wealth_ : ndarray of shape (T+1,)
        Wealth history including the initial value, followed by wealth after each period.

    all_weights_ : ndarray of shape (T, n_assets)
        Weight history: weights used at the start of each period (before observing returns).
    """

    n_features_in_: int

    def __init__(
        self,
        weights: npt.ArrayLike | None,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(portfolio_params=portfolio_params)
        self.weights = weights
        self.portfolio_params = portfolio_params

    def partial_fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> ConstantRebalancedPortfolio:
        """Update the comparator target using the supplied observations."""
        net_to_relatives(X)
        self._set_weights(X)
        return self

    def _set_weights(self, X: npt.ArrayLike) -> None:
        """Set finite fixed weights matching the validated asset count."""
        n_assets = np.shape(X)[1]
        if self.weights is None:
            self.weights_ = np.asarray(EqualWeighted().fit(X).weights_, dtype=float)
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
        if self.weights_.shape != (n_assets,) or not np.all(np.isfinite(self.weights_)):
            raise ValueError("weights must contain one finite value per asset")

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> ConstantRebalancedPortfolio:
        """Set fixed weights and compute the gross wealth trajectory.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self
            The estimator instance.
        """
        relatives = net_to_relatives(X)
        self._set_weights(X)
        self.all_wealth_, self.all_weights_ = _constant_wealth_history(
            relatives=relatives, weights=self.weights_, initial_wealth=1.0
        )
        self.wealth_ = float(self.all_wealth_[-1])
        return self


class BestAsset(BaseOptimization):
    """Best asset (ex-post).

    Invests all capital in the single asset with highest cumulative log-return.
    This is a hindsight benchmark showing the best possible single-asset performance.

    Parameters
    ----------
    portfolio_params : dict, optional
        Portfolio parameters propagated to the produced portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Portfolio weights (one-hot vector with 1.0 on best asset).

    wealth_ : float
        Gross hindsight wealth after all periods, excluding costs and fees.

    all_wealth_ : ndarray of shape (T+1,)
        Wealth history including the initial value, followed by wealth after each period.

    all_weights_ : ndarray of shape (T, n_assets)
        Weight history: weights used at the start of each period (before observing returns).
    """

    def __init__(self, portfolio_params: dict | None = None) -> None:
        super().__init__(portfolio_params=portfolio_params)

    def partial_fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> BestAsset:
        """Update the comparator target using the supplied observations."""
        self._set_weights(net_to_relatives(X))
        return self

    def _set_weights(self, relatives: np.ndarray) -> None:
        """Select the asset with the highest cumulative log return."""
        best_asset = np.argmax(np.sum(np.log(relatives), axis=0))
        self.weights_ = np.zeros(relatives.shape[1])
        self.weights_[best_asset] = 1.0

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> BestAsset:
        """Select the best asset and compute its gross wealth trajectory.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self
            The estimator instance.
        """
        relatives = net_to_relatives(X)
        self._set_weights(relatives)
        self.all_wealth_, self.all_weights_ = _constant_wealth_history(
            relatives=relatives, weights=self.weights_, initial_wealth=1.0
        )
        self.wealth_ = float(self.all_wealth_[-1])
        return self
