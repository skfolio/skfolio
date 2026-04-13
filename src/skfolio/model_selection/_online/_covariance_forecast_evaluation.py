"""Online covariance forecast evaluation."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sklearn as sk
import sklearn.base as skb
import sklearn.utils as sku
from sklearn.pipeline import Pipeline

from skfolio.metrics._covariance import _get_covariance
from skfolio.model_selection._covariance_forecast_evaluation import (
    CovarianceForecastEvaluation,
    _compute_step_diagnostics,
    _normalize_portfolio_weights,
)
from skfolio.model_selection._online._validation import (
    _online_walk_forward,
    _route_params,
    _validate_online_estimator,
    _validate_sizes,
)
from skfolio.utils.tools import safe_indexing


def online_covariance_forecast_evaluation(
    estimator: skb.BaseEstimator | Pipeline,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    warmup_size: int = 252,
    test_size: int = 1,
    portfolio_weights: npt.ArrayLike | None = None,
    purged_size: int = 0,
    params: dict | None = None,
) -> CovarianceForecastEvaluation:
    r"""Evaluate out-of-sample covariance forecast quality.

    Walks forward through the data using incremental learning and computes per-step
    calibration diagnostics comparing the covariance forecast to realized returns.

    At each step the estimator is updated via `partial_fit` and the fitted covariance is
    evaluated against the next `test_size` observations. This is the online counterpart
    of :func:`~skfolio.model_selection.covariance_forecast_evaluation`, which instead
    refits the estimator from scratch on each training window.

    Every evaluation window contains exactly `test_size` observations, ensuring that
    diagnostics (in particular QLIKE) are directly comparable across steps.

    Four core diagnostics are computed:

    * **Mahalanobis calibration ratio**: tests whether the full covariance
      structure (all eigenvalue directions) is correctly specified. The
      target is 1.0. A value above 1.0 indicates underestimated risk;
      below 1.0 indicates overestimated risk.
    * **Diagonal calibration ratio**: tests whether the individual asset
      variances are correctly specified, ignoring correlations. The target
      is 1.0. A value above 1.0 indicates underestimated volatilities;
      below 1.0 indicates overestimated volatilities.
    * **Portfolio standardized returns / bias statistic**: tests whether
      the covariance is well calibrated along one or more portfolio
      directions.
    * **Portfolio QLIKE**: evaluates portfolio variance forecasts along
      one or more portfolio directions by comparing the forecast portfolio
      variance with the realized sum of squared portfolio returns over the
      evaluation window. Lower values indicate better portfolio variance
      forecasts.

    When the test returns contain NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), only finite observations contribute to the aggregated return. For
    portfolio diagnostics, NaN returns for active assets contribute zero to the realized
    portfolio return and the forecast covariance is scaled by the pairwise observation
    count matrix :math:`H` (Hadamard product :math:`H \odot \Sigma`) so that the
    realized portfolio variance and forecast variance follow the same missing-data
    convention. In skfolio, NaN diagonal entries in the forecast covariance mark
    inactive assets, which are excluded from the evaluation.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted estimator or Pipeline. Must expose `covariance_` or
        `return_distribution_.covariance` after fitting.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : Ignored
        Present for scikit-learn API compatibility.

    warmup_size : int, default=252
        Number of initial observations used for the first `partial_fit` call.

    test_size : int, default=1
        Number of observations per evaluation window. All windows have exactly this many
        observations.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        Portfolio weights for portfolio-level diagnostics (bias statistic and QLIKE).

        If `None` (default), inverse-volatility weights are used, recomputed dynamically
        at each step from the forecast covariance. This neutralizes volatility
        dispersion so that high-volatility assets do not dominate the diagnostic.

        If a 1D array is provided, a single static portfolio is used.

        If a 2D array of shape `(n_portfolios, n_assets)` is provided, each row defines
        a test portfolio and diagnostics are computed independently for each.

        For equal-weight calibration, pass `portfolio_weights=np.ones(n_assets) / n_assets`.

    purged_size : int, default=0
        Number of observations to skip between training and test data.

    params : dict, optional
        Parameters routed to the estimator's `partial_fit` via metadata routing.

    Returns
    -------
    evaluation : CovarianceForecastEvaluation
        Frozen dataclass with per-step calibration arrays, summary statistics, and
        plotting methods.

    Raises
    ------
    TypeError
        If the estimator does not support `partial_fit`.

    ValueError
        If the data is too short for at least one evaluation step.

    See Also
    --------
    covariance_forecast_evaluation : Batch counterpart that refits the
        estimator from scratch on each training window.
    CovarianceForecastEvaluation : Result dataclass with summary statistics
        and plotting methods.
    :ref:`sphx_glr_auto_examples_online_learning_plot_1_online_covariance_forecast_evaluation.py`
        End-to-end covariance forecast evaluation tutorial.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import (
    ...     online_covariance_forecast_evaluation,
    ... )
    >>> from skfolio.moments import EWCovariance
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> evaluation = online_covariance_forecast_evaluation(  # doctest: +SKIP
    ...     EWCovariance(half_life=60),
    ...     X,
    ...     warmup_size=252,
    ...     test_size=5,
    ... )
    >>> evaluation.summary()  # doctest: +SKIP
    >>> evaluation.bias_statistic  # doctest: +SKIP
    >>> evaluation.plot_calibration()  # doctest: +SKIP
    """
    _validate_online_estimator(
        estimator,
        caller="online_covariance_forecast_evaluation",
    )
    _validate_sizes(warmup_size, test_size)

    estimator = sk.clone(estimator)
    X, y = sku.indexable(X, y)
    observations = X.index if hasattr(X, "index") else np.arange(len(X))
    routed_params = _route_params(
        estimator,
        params,
        owner="online_covariance_forecast_evaluation",
        callee="partial_fit",
    )

    portfolio_weights = _normalize_portfolio_weights(portfolio_weights)
    n_portfolios = portfolio_weights.shape[0] if portfolio_weights is not None else 1

    evaluation_observations: list = []
    squared_mahalanobis_distances: list[float] = []
    mahalanobis_calibration_ratios: list[float] = []
    diagonal_calibration_ratios: list[float] = []
    portfolio_standardized_returns: list[np.ndarray] = []
    portfolio_variance_qlike_losses: list[np.ndarray] = []
    valid_asset_counts: list[int] = []

    for test_slice in _online_walk_forward(
        estimator,
        X,
        y,
        warmup_size,
        test_size,
        routed_params,
        purged_size=purged_size,
    ):
        covariance = _get_covariance(estimator)

        step = _compute_step_diagnostics(
            covariance, safe_indexing(X, indices=test_slice), portfolio_weights
        )
        if step is None:
            continue

        (
            squared_mahalanobis_distance,
            mahalanobis_calibration_ratio,
            diagonal_calibration_ratio,
            standardized_portfolio_return,
            portfolio_variance_qlike_loss,
            n_valid_assets,
        ) = step

        squared_mahalanobis_distances.append(squared_mahalanobis_distance)
        mahalanobis_calibration_ratios.append(mahalanobis_calibration_ratio)
        diagonal_calibration_ratios.append(diagonal_calibration_ratio)
        portfolio_standardized_returns.append(standardized_portfolio_return)
        portfolio_variance_qlike_losses.append(portfolio_variance_qlike_loss)
        valid_asset_counts.append(n_valid_assets)
        evaluation_observations.append(observations[test_slice.start])

    return CovarianceForecastEvaluation(
        observations=np.array(evaluation_observations),
        horizon=test_size,
        squared_mahalanobis_distance=np.array(squared_mahalanobis_distances),
        mahalanobis_calibration_ratio=np.array(mahalanobis_calibration_ratios),
        diagonal_calibration_ratio=np.array(diagonal_calibration_ratios),
        portfolio_standardized_return=np.array(portfolio_standardized_returns),
        portfolio_variance_qlike_loss=np.array(portfolio_variance_qlike_losses),
        n_valid_assets=np.array(valid_asset_counts, dtype=int),
        n_portfolios=n_portfolios,
        name=str(estimator),
    )
