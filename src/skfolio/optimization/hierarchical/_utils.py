"""Shared distance fitting, risk calculations and hierarchical weight bounds."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn.utils.metadata_routing as skm
from sklearn import get_config

from skfolio._constants import _MANAGEMENT_FEES, _PREVIOUS_WEIGHTS, _TRANSACTION_COSTS
from skfolio.distance import BaseDistance
from skfolio.exceptions import OptimizationError
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.portfolio import Portfolio
from skfolio.prior import ReturnDistribution
from skfolio.typing import AnyArray, FloatArray, IntArray, MultiInput
from skfolio.utils.tools import _filter_supported_params

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
    """Broadcast and validate bounds before selecting the investable assets.

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
    A full-universe upper sum below one is infeasible for every subset and is rejected
    here, allowing invalid inputs to fail before online learning. Lower sums are checked
    by `_check_investable_weight_bounds` after selection: excluded assets do not
    contribute to the allocation's minimum weights. Bound sums allow an absolute
    tolerance of 1e-8 around one.
    """
    if not isinstance(min_weights, np.ndarray):
        min_weights = np.full(n_assets, min_weights)
    if np.any(min_weights < 0):
        raise ValueError("`min_weights` must be strictly positive")

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


def _check_investable_weight_bounds(
    *, min_weights: FloatArray, max_weights: FloatArray
) -> None:
    """Check that investable asset weights can sum to one.

    Parameters
    ----------
    min_weights : FloatArray of shape (n_assets,)
        Lower weight bounds restricted to the investable assets.

    max_weights : FloatArray of shape (n_assets,)
        Upper weight bounds restricted to the investable assets.

    Raises
    ------
    OptimizationError
        If the lower bounds sum to more than one or the upper bounds sum to less than
        one, allowing an absolute tolerance of 1e-8.
    """
    min_weights_sum = min_weights.sum()
    max_weights_sum = max_weights.sum()
    if (
        min_weights_sum > 1 + _WEIGHT_BOUNDS_TOL
        or max_weights_sum < 1 - _WEIGHT_BOUNDS_TOL
    ):
        raise OptimizationError(
            "Weight bounds are infeasible for the investable assets: "
            f"lower bounds sum to {min_weights_sum:.12g} and upper bounds sum to "
            f"{max_weights_sum:.12g}. The lower sum must be at most 1 and the upper "
            f"sum at least 1, within a tolerance of {_WEIGHT_BOUNDS_TOL:g}."
        )


def _check_investable_distances(
    distance: FloatArray,
    *,
    n_assets: int,
    asset_indices: IntArray,
    asset_names: AnyArray | None = None,
) -> None:
    """Validate distances for the investable asset universe.

    Parameters
    ----------
    distance : ndarray of shape (n_assets, n_assets)
        Distances between the assets selected for allocation.

    n_assets : int
        Expected number of investable assets, independent of input labels.

    asset_indices : ndarray of shape (n_assets,)
        Zero-based column positions in the optimizer's original `X`, used in error
        messages when asset names are unavailable.

    asset_names : ndarray of shape (n_assets,), optional
        Asset labels in distance order, used to identify unavailable estimates in the
        error message. None for unlabeled input.

    Raises
    ------
    ValueError
        If the matrix has an unexpected shape or contains non-finite distances.
    """
    if distance.shape != (n_assets, n_assets):
        raise ValueError("The distance output does not match its input asset schema.")

    finite = np.isfinite(distance)
    if finite.all():
        return

    # A missing row also invalidates every counterparty's distance to that asset.
    unavailable = ~finite.any(axis=1) | ~finite.any(axis=0)
    if not unavailable.any():
        unavailable = ~finite.all(axis=1) | ~finite.all(axis=0)

    message = "Distances are unavailable for prior-investable assets"
    if asset_names is not None:
        message += f" {asset_names[unavailable].tolist()}"
    else:
        message += f" at X column positions {asset_indices[unavailable].tolist()}"
    raise ValueError(
        f"{message}. Check the distance estimator's data coverage and "
        "minimum-observation requirements."
    )


def _fit_distance_from_prior(
    *,
    estimator: BaseDistance,
    return_distribution: ReturnDistribution,
    asset_names: AnyArray | None,
    fit_params: dict[str, Any],
) -> FloatArray:
    """Fit distances on a prepared prior distribution without cloning the estimator.

    Parameters
    ----------
    estimator : BaseDistance
        Distance estimator to fit on returns or covariance.

    return_distribution : ReturnDistribution
        Distribution restricted to the assets used for allocation.

    asset_names : ndarray or None
        Names in distribution order, or None for unlabeled inputs.

    fit_params : dict
        Routed fitting metadata. Requested scenario weights are supplied by the
        distribution, replacing weights for the optimizer's original input. Covariance
        input accepts no observation metadata.

    Returns
    -------
    distance : ndarray of shape (n_assets, n_assets)
        Fitted distances in distribution order.
    """
    requires_covariance_input = estimator.requires_covariance_input
    fit_params = fit_params.copy()
    if requires_covariance_input:
        if fit_params:
            raise ValueError(
                "Observation metadata cannot be passed to a distance estimator "
                "fitted on the prior's covariance, whose rows represent assets."
            )
        distance_input = return_distribution.covariance
    else:
        distance_input = return_distribution.returns
        if get_config()["enable_metadata_routing"]:
            fit_params.update(
                dict.fromkeys(
                    _sample_weight_names(estimator.get_metadata_routing()),
                    return_distribution.sample_weight,
                )
            )
        else:
            fit_params.update(
                _filter_supported_params(
                    estimator, "fit", sample_weight=return_distribution.sample_weight
                )
            )
    if asset_names is not None:
        distance_input = pd.DataFrame(
            distance_input,
            columns=asset_names,
            index=asset_names if requires_covariance_input else None,
        )
    estimator.fit(distance_input, None, **fit_params)
    return estimator.distance_


def _sample_weight_names(
    routing: skm.MetadataRequest | skm.MetadataRouter,
    *,
    method: str = "fit",
    use_alias: bool = False,
) -> set[str]:
    """Find the keyword names that deliver sample weights to an estimator.

    Parameters
    ----------
    routing : MetadataRequest or MetadataRouter
        Routing description of the distance estimator or one of its children.

    method : str, default="fit"
        Method receiving the weights. Recursive calls follow the router's method
        mappings.

    use_alias : bool, default=False
        Use the name requested from the parent. A direct call to a consumer uses
        `sample_weight`, while a router receives the aliases requested by its
        children.

    Returns
    -------
    names : set of str
        Keywords requested for sample weights. Unrequested weights are excluded.
    """
    if isinstance(routing, skm.MetadataRequest):
        request = getattr(routing, method).requests.get("sample_weight")
        if request in (None, False, skm.WARN):
            return set()
        if request is True or not use_alias:
            return {"sample_weight"}
        return {request}

    names = set()
    for name, route in routing:
        for caller, callee in route.mapping:
            if caller == method:
                # A router's own consumer keeps the current naming rule.
                # Descendants request their weights from a parent using aliases.
                names.update(
                    _sample_weight_names(
                        route.router,
                        method=callee,
                        use_alias=use_alias or name != "$self_request",
                    )
                )
    return names
