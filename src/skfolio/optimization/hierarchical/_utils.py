"""Shared risk calculations and weight bounds for hierarchical optimizers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio._constants import _MANAGEMENT_FEES, _PREVIOUS_WEIGHTS, _TRANSACTION_COSTS
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.portfolio import Portfolio
from skfolio.prior import ReturnDistribution
from skfolio.typing import FloatArray, IntArray, MultiInput

_WEIGHT_BOUNDS_TOL = 1e-8


class _PortfolioRiskMixin:
    """Compute portfolio risk and the risk of investing in each asset separately.

    Shared by HRP and HERC. This mixin requires a `BaseOptimization` subclass
    that defines `risk_measure`, `transaction_costs` and `management_fees`.
    It uses `_clean_input` and `_clean_previous_weights` from `BaseOptimization`
    to align costs and previous holdings with the assets in the return distribution.
    """

    risk_measure: RiskMeasure | ExtraRiskMeasure
    transaction_costs: MultiInput
    management_fees: MultiInput

    def _risk(
        self,
        weights: FloatArray,
        return_distribution: ReturnDistribution,
    ) -> float:
        """Compute the configured risk measure for the supplied portfolio weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights, aligned with the assets in `return_distribution`.

        return_distribution : ReturnDistribution
            Asset returns, covariance and optional scenario weights.

        Returns
        -------
        risk : float
            Portfolio risk evaluated using `risk_measure`.

        Raises
        ------
        ValueError
            If costs, fees or previous holdings have invalid shapes or non-finite
            values for the assets in `return_distribution`.
        """
        n_assets = return_distribution.returns.shape[1]
        transaction_costs = self._clean_input(  # ty: ignore[unresolved-attribute]
            self.transaction_costs,
            n_assets=n_assets,
            fill_value=0,
            name=_TRANSACTION_COSTS,
        )
        management_fees = self._clean_input(  # ty: ignore[unresolved-attribute]
            self.management_fees,
            n_assets=n_assets,
            fill_value=0,
            name=_MANAGEMENT_FEES,
        )
        previous_weights = self._clean_previous_weights(  # ty: ignore[unresolved-attribute]
            n_assets=n_assets
        )
        # Invalid inputs must raise before they can produce a non-finite risk.
        for name, value in (
            (_TRANSACTION_COSTS, transaction_costs),
            (_MANAGEMENT_FEES, management_fees),
            (_PREVIOUS_WEIGHTS, previous_weights),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite.")

        ptf = Portfolio(
            X=return_distribution.returns,
            sample_weight=return_distribution.sample_weight,
            weights=weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
        )
        if self.risk_measure in [RiskMeasure.VARIANCE, RiskMeasure.STANDARD_DEVIATION]:
            risk = ptf.variance_from_assets(
                assets_covariance=return_distribution.covariance
            )
            if self.risk_measure == RiskMeasure.STANDARD_DEVIATION:
                risk = np.sqrt(risk)
        else:
            risk = getattr(ptf, str(self.risk_measure.value))
        return risk

    def _unitary_risks(self, return_distribution: ReturnDistribution) -> FloatArray:
        """Compute the risk of a portfolio invested entirely in each asset.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            Asset returns, covariance and optional scenario weights.

        Returns
        -------
        risks : ndarray of shape (n_assets,)
            Risk from allocating all capital to each asset, in distribution order.
        """
        n_assets = return_distribution.returns.shape[1]
        risks = [
            self._risk(weights=weights, return_distribution=return_distribution)
            for weights in np.identity(n_assets)
        ]
        return np.array(risks)


def _convert_weight_bounds(
    min_weights: float | FloatArray,
    max_weights: float | FloatArray,
    n_assets: int,
) -> tuple[FloatArray, FloatArray]:
    """Broadcast cleaned bounds and validate long-only, fully invested weights.

    Parameters
    ----------
    min_weights : float or ndarray of shape (n_assets,)
        Lower bounds after resolving asset names and missing values.

    max_weights : float or ndarray of shape (n_assets,)
        Upper bounds after resolving asset names and missing values.

    n_assets : int
        Number of assets in the full input universe.

    Returns
    -------
    min_weights : ndarray of shape (n_assets,)
        Lower weight bounds.

    max_weights : ndarray of shape (n_assets,)
        Upper weight bounds.

    Notes
    -----
    Bound sums allow an absolute tolerance of 1e-8 around one.
    """
    if not isinstance(min_weights, np.ndarray):
        min_weights = np.full(n_assets, min_weights)
    if np.any(min_weights < 0):
        raise ValueError("`min_weights` must be strictly positive")
    if min_weights.sum() > 1 + _WEIGHT_BOUNDS_TOL:
        raise ValueError(
            f"Invalid `min_weights`: sum is {min_weights.sum():.4f}, "
            f"but it must be at most 1.0."
        )

    if not isinstance(max_weights, np.ndarray):
        max_weights = np.full(n_assets, max_weights)
    if np.any(max_weights > 1):
        raise ValueError("`max_weights` must be less than or equal to 1.0")
    if max_weights.sum() < 1 - _WEIGHT_BOUNDS_TOL:
        raise ValueError(
            f"Invalid `max_weights`: sum is {max_weights.sum():.4f}, "
            f"but it must be at least 1.0."
        )
    if not np.isfinite(min_weights).all() or not np.isfinite(max_weights).all():
        raise ValueError("Weight bounds must be finite.")
    if np.any(min_weights > max_weights):
        raise NameError(
            "Items of `min_weights` must be less than or equal to items of"
            " `max_weights`"
        )
    return min_weights, max_weights


def _apply_weight_constraints_to_split_factor(
    alpha: float,
    max_weights: FloatArray,
    min_weights: FloatArray,
    weights: FloatArray,
    left_cluster: IntArray,
    right_cluster: IntArray,
) -> float:
    """Apply weight bounds to a recursive allocation split.

    Parameters
    ----------
    alpha : float
        The split factor alpha of the Hierarchical Tree Clustering algorithm.

    min_weights : ndarray of shape (n_assets,)
        The weight lower bound 1D array.

    max_weights : ndarray of shape (n_assets,)
        The weight upper bound 1D array.

    weights : FloatArray of shape (n_assets,)
        The assets weights.

    left_cluster : ndarray of shape (n_left_cluster,)
        Indices of the left cluster weights.

    right_cluster : ndarray of shape (n_right_cluster,)
        Indices of the right cluster weights.

    Returns
    -------
    value : float
        The transformed split factor alpha incorporating the weight constraints.
    """
    alpha = min(
        np.sum(max_weights[left_cluster]) / weights[left_cluster[0]],
        max(np.sum(min_weights[left_cluster]) / weights[left_cluster[0]], alpha),
    )
    alpha = 1 - min(
        np.sum(max_weights[right_cluster]) / weights[right_cluster[0]],
        max(
            np.sum(min_weights[right_cluster]) / weights[right_cluster[0]],
            1 - alpha,
        ),
    )
    return alpha
