"""Metrics for evaluating covariance forecasts."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipy.stats as sst
import sklearn.base as skb

from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import inverse_volatility_weights, squared_mahalanobis_dist

__all__ = [
    "diagonal_calibration_loss",
    "diagonal_calibration_ratio",
    "exceedance_rate",
    "mahalanobis_calibration_loss",
    "mahalanobis_calibration_ratio",
    "portfolio_variance_calibration_loss",
    "portfolio_variance_calibration_ratio",
    "portfolio_variance_qlike_loss",
    "qlike_loss",
]

_NUMERICAL_THRESHOLD = 1e-12


def mahalanobis_calibration_ratio(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
) -> float:
    r"""Mahalanobis calibration ratio.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`,
    and let :math:`R^{(h)} = \sum_{t=1}^{h} r_t` be the aggregated return over
    an evaluation window of :math:`h` observations. This metric compares
    :math:`R^{(h)}` against the horizon-scaled covariance :math:`h\,\Sigma`:

    .. math::

        s = \frac{{R^{(h)}}^\top (h\,\Sigma)^{-1} R^{(h)}}{n}

    where :math:`n` is the number of assets.

    If the forecast covariance is correct and the aggregated return is centered,
    then :math:`\mathbb{E}[s] = 1` for any horizon :math:`h`. Under
    multivariate normality, :math:`n s \sim \chi^2(n)`.

    For financial return series, heavy tails and regime changes can cause
    departures from the Gaussian reference. In practice, this ratio is often
    most useful as a relative diagnostic across estimators.

    When `X_test` contains NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), only finite observations are used in the aggregated return and
    the covariance is scaled by the pairwise observation count matrix
    :math:`H` (Hadamard product :math:`H \odot \Sigma`) so that the same
    target applies with missing data. In skfolio, NaN diagonal entries in the
    forecast covariance mark inactive assets, which are excluded from the
    evaluation.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    Returns
    -------
    float
        Calibration ratio. Values near `1.0` indicate that the forecast
        covariance matches the scale of the realized aggregated return.

    See Also
    --------
    mahalanobis_calibration_loss : Absolute deviation from the calibration
        target of `1.0`.
    diagonal_calibration_ratio : Calibration ratio using only marginal
        variances.
    """
    cov = _get_covariance(estimator)
    result = _prepare_active_subset(cov, X_test)
    if result is None:
        return float(np.nan)
    active_cov, active_returns, _, n_valid_assets = result
    aggregated_return, effective_cov = _aggregated_return_and_effective_covariance(
        active_returns, active_cov
    )
    d2 = squared_mahalanobis_dist(aggregated_return, effective_cov)
    return float(d2 / n_valid_assets)


def diagonal_calibration_ratio(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
) -> float:
    r"""Diagonal calibration ratio based on marginal variances.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`,
    and let :math:`R^{(h)} = \sum_{t=1}^{h} r_t` be the aggregated return over
    an evaluation window of :math:`h` observations. This metric uses only the
    diagonal of the covariance matrix, ignoring correlations, and compares
    each component :math:`R_i^{(h)}` against the horizon-scaled variance
    :math:`h\,\sigma_i^2`:

    .. math::

        s = \frac{1}{n}\sum_{i=1}^{n}
            \frac{(R_i^{(h)})^2}{h\,\sigma_i^2}

    where :math:`n` is the number of assets and :math:`\sigma_i^2` is the
    forecast variance for asset :math:`i`.

    If the marginal variance forecasts are correct and the aggregated returns are
    centered, then :math:`\mathbb{E}[s] = 1` for any horizon :math:`h`.
    Because correlations are ignored, this metric diagnoses the calibration
    of marginal scales rather than the full covariance structure.

    When `X_test` contains NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), each asset uses its own effective horizon :math:`h_i`, equal to
    the number of finite observations for that asset, so the ratio retains the
    same target under missing data. In skfolio, NaN diagonal entries in the
    forecast covariance mark inactive assets, which are excluded from the
    evaluation.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    Returns
    -------
    float
        Calibration ratio. Values near `1.0` indicate well-calibrated
        marginal variance forecasts.

    See Also
    --------
    diagonal_calibration_loss : Absolute deviation from the calibration
        target of `1.0`.
    mahalanobis_calibration_ratio : Calibration ratio using the full
        covariance structure.
    """
    cov = _get_covariance(estimator)
    result = _prepare_active_subset(cov, X_test)
    if result is None:
        return float(np.nan)
    active_cov, active_returns, _, n_valid_assets = result
    active_mask = np.isfinite(active_returns)
    effective_horizon_per_asset = active_mask.sum(axis=0)
    aggregated_return = np.where(active_mask, active_returns, 0.0).sum(axis=0)
    scaled_var = effective_horizon_per_asset * np.maximum(
        np.diag(active_cov), _NUMERICAL_THRESHOLD
    )
    return float(np.sum(aggregated_return**2 / scaled_var) / n_valid_assets)


def portfolio_variance_calibration_ratio(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
    portfolio_weights: ArrayLike | None = None,
) -> float:
    r"""Portfolio variance calibration ratio.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`
    and :math:`w^\top r_t` the corresponding one-period portfolio return for
    weights :math:`w`. This metric compares the sum of squared portfolio
    returns over an evaluation window of :math:`h` observations to the
    horizon-scaled forecast portfolio variance:

    .. math::

        s = \frac{\sum_{t=1}^{h} (w^\top r_t)^2}
                 {h\, w^\top \Sigma\, w}

    If the projected portfolio variance is correctly specified and portfolio
    returns are centered, then :math:`\mathbb{E}[s] = 1`.

    When `X_test` contains NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), NaN returns for active assets contribute zero to the realized
    portfolio return. The forecast covariance is scaled by the pairwise
    observation count matrix :math:`H` (Hadamard product
    :math:`H \odot \Sigma`) so that the realized portfolio variance and
    forecast variance follow the same missing-data convention. In skfolio,
    NaN diagonal entries in the forecast covariance mark inactive assets,
    which are excluded before the score is computed.

    When multiple portfolios are provided (2D weights), the ratio is computed
    independently for each and the mean is returned. This produces a more
    robust diagnostic by averaging across multiple portfolio directions.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        Portfolio weights. If `None` (default), inverse-volatility weights are used,
        which neutralizes volatility dispersion so that high-volatility assets do not
        dominate the diagnostic. If a 2D array is provided, each row defines a test
        portfolio and the mean ratio across portfolios is returned. For equal-weight
        calibration, pass `portfolio_weights=np.ones(n_assets) / n_assets`.

    Returns
    -------
    float
        Calibration ratio. Values near `1.0` indicate that the projected
        portfolio variance is well calibrated on average.

    See Also
    --------
    portfolio_variance_calibration_loss : Absolute deviation from the
        calibration target of `1.0`.
    portfolio_variance_qlike_loss : QLIKE loss for the projected portfolio
        variance.
    """
    cov = _get_covariance(estimator)
    result = _prepare_active_subset(cov, X_test)
    if result is None:
        return float(np.nan)
    active_cov, active_returns, active_asset_indices, _ = result
    w = _resolve_weights(active_cov, portfolio_weights, active_asset_indices)
    _, effective_cov = _aggregated_return_and_effective_covariance(
        active_returns, active_cov
    )
    forecast_var = np.maximum(
        np.sum(w * (w @ effective_cov), axis=1), _NUMERICAL_THRESHOLD
    )
    active_returns_filled = np.where(np.isfinite(active_returns), active_returns, 0.0)
    realized_ptf = active_returns_filled @ w.T
    realized_var = np.sum(realized_ptf**2, axis=0)
    return float(np.mean(realized_var / forecast_var))


def mahalanobis_calibration_loss(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
) -> float:
    r"""Mahalanobis calibration loss.

    Computes the absolute deviation of :func:`mahalanobis_calibration_ratio` from its
    calibration target of `1.0`.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`,
    and let :math:`R^{(h)} = \sum_{t=1}^{h} r_t` be the aggregated return over
    an evaluation window of :math:`h` observations.

    .. math::

        \ell = \left\lvert
            \frac{{R^{(h)}}^\top (h\,\Sigma)^{-1} R^{(h)}}{n} - 1
        \right\rvert

    where :math:`n` is the number of assets.

    As with :func:`mahalanobis_calibration_ratio`, heavy tails and regime
    changes can weaken the Gaussian reference. This loss is therefore often
    most useful for relative comparison across covariance estimators.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    Returns
    -------
    float
        Calibration loss. Lower values are better and the optimum is `0.0`.

    See Also
    --------
    mahalanobis_calibration_ratio : The underlying calibration ratio.
    diagonal_calibration_loss : Loss using only marginal variances.
    """
    return abs(mahalanobis_calibration_ratio(estimator, X_test, y) - 1.0)


def diagonal_calibration_loss(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
) -> float:
    r"""Diagonal calibration loss.

    Computes the absolute deviation of :func:`diagonal_calibration_ratio` from its
    calibration target of `1.0`.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`,
    and let :math:`R^{(h)} = \sum_{t=1}^{h} r_t` be the aggregated return over
    an evaluation window of :math:`h` observations.

    .. math::

        \ell = \left\lvert
            \frac{1}{n}\sum_{i=1}^{n}
            \frac{(R_i^{(h)})^2}{h\,\sigma_i^2} - 1
        \right\rvert

    where :math:`n` is the number of assets.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    Returns
    -------
    float
        Calibration loss. Lower values are better and the optimum is `0.0`.

    See Also
    --------
    diagonal_calibration_ratio : The underlying calibration ratio.
    mahalanobis_calibration_loss : Loss using the full covariance structure.
    """
    return abs(diagonal_calibration_ratio(estimator, X_test, y) - 1.0)


def portfolio_variance_calibration_loss(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
    portfolio_weights: ArrayLike | None = None,
) -> float:
    r"""Portfolio variance calibration loss.

    Computes the absolute deviation of :func:`portfolio_variance_calibration_ratio` from
    its calibration target of `1.0`.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`
    and :math:`w^\top r_t` the corresponding one-period portfolio return for
    weights :math:`w`.

    .. math::

        \ell = \left\lvert
            \frac{\sum_{t=1}^{h} (w^\top r_t)^2}
                 {h\, w^\top \Sigma\, w} - 1
        \right\rvert

    When multiple portfolios are provided, the loss is the absolute deviation
    of the mean ratio from `1.0`.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        Portfolio weights. If `None` (default), inverse-volatility weights are used,
        which neutralizes volatility dispersion so that high-volatility assets do not
        dominate the diagnostic. If a 2D array is provided, each row defines a test
        portfolio. For equal-weight calibration, pass
        `portfolio_weights=np.ones(n_assets) / n_assets`.

    Returns
    -------
    float
        Calibration loss. Lower values are better and the optimum is `0.0`.

    See Also
    --------
    portfolio_variance_calibration_ratio : The underlying calibration ratio.
    portfolio_variance_qlike_loss : QLIKE loss for the projected portfolio
        variance.
    """
    return abs(
        portfolio_variance_calibration_ratio(estimator, X_test, y, portfolio_weights)
        - 1.0
    )


def portfolio_variance_qlike_loss(
    estimator: skb.BaseEstimator,
    X_test: ArrayLike,
    y=None,
    portfolio_weights: ArrayLike | None = None,
) -> float:
    r"""QLIKE loss for a projected portfolio variance forecast [1]_.

    Let :math:`r_t` be the one-period realized return vector at time :math:`t`
    and :math:`w^\top r_t` the corresponding one-period portfolio return for
    weights :math:`w`. The loss compares the forecast portfolio variance
    with the realized sum of squared portfolio returns over the evaluation
    window of :math:`h` observations:

    .. math::

        \ell = \log\left(h\, w^\top \Sigma\, w\right)
             + \frac{\sum_{t=1}^{h} (w^\top r_t)^2}{h\, w^\top \Sigma\, w}

    Lower values are better. In expectation, the loss is minimized by the
    true conditional portfolio variance forecast.

    When `X_test` contains NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), NaN returns for active assets contribute zero to the realized
    portfolio return. The forecast covariance is scaled by the pairwise
    observation count matrix :math:`H` (Hadamard product
    :math:`H \odot \Sigma`) so that the realized portfolio variance and
    forecast variance follow the same missing-data convention. In skfolio,
    NaN diagonal entries in the forecast covariance mark inactive assets,
    which are excluded before the score is computed.

    When multiple portfolios are provided (2D weights), the QLIKE is computed
    independently for each and the mean is returned. This lets one summary
    score evaluate several portfolio directions at once.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator, must expose `covariance_` or `return_distribution_.covariance`.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns for the test window.

    y : Ignored
        Present for scikit-learn API compatibility.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        Portfolio weights. If `None` (default), inverse-volatility weights are used,
        which neutralizes volatility dispersion so that high-volatility assets do not
        dominate the diagnostic. If a 2D array is provided, each row defines a test
        portfolio and the mean QLIKE across portfolios is returned.

    Returns
    -------
    float
        Mean portfolio QLIKE loss. Lower values are better; in expectation, the loss
        is minimized by the true conditional portfolio variance forecast.

    See Also
    --------
    portfolio_variance_calibration_ratio : Calibration ratio for the
        projected portfolio variance.
    portfolio_variance_calibration_loss : Calibration loss for the projected
        portfolio variance.
    qlike_loss : Univariate QLIKE loss.

    References
    ----------
    .. [1] "Volatility forecast comparison using imperfect volatility proxies"
        Journal of Econometrics. Patton, A. J. (2011).
    """
    cov = _get_covariance(estimator)
    result = _prepare_active_subset(cov, X_test)
    if result is None:
        return float(np.nan)
    active_cov, active_returns, active_asset_indices, _ = result
    w = _resolve_weights(active_cov, portfolio_weights, active_asset_indices)
    _, effective_cov = _aggregated_return_and_effective_covariance(
        active_returns, active_cov
    )
    forecast_var = np.maximum(
        np.sum(w * (w @ effective_cov), axis=1), _NUMERICAL_THRESHOLD
    )
    active_returns_filled = np.where(np.isfinite(active_returns), active_returns, 0.0)
    realized_ptf = active_returns_filled @ w.T
    realized_var = np.sum(realized_ptf**2, axis=0)
    return float(np.mean(np.log(forecast_var) + realized_var / forecast_var))


def exceedance_rate(
    squared_distances: ArrayLike,
    n_features: int,
    confidence_level: float,
) -> float:
    r"""Exceedance rate for chi-squared calibration statistics.

    Computes the fraction of squared distances exceeding the upper `confidence_level`
    chi-squared quantile.

    The reference threshold assumes Gaussian standardized returns. In practice,
    the rate is sensitive not only to covariance misspecification but also to
    heavy tails, regime shifts, and non-Gaussian standardized returns. It is
    best used as a comparative metric across estimators rather than as
    an absolute calibration test.

    Parameters
    ----------
    squared_distances : array-like of shape (n_observations,)
        Squared Mahalanobis distances or similar chi-squared statistics.

    n_features : int
        Degrees of freedom (number of features/assets).

    confidence_level : float
        Coverage confidence level used to define the upper chi-squared
        threshold. For example, `0.95` corresponds to an expected
        exceedance rate of `0.05` under calibration.

    Returns
    -------
    float
        Observed exceedance rate. It should be close to
        :math:`1 - \text{confidence\_level}` when the reference
        chi-squared approximation is appropriate.

    See Also
    --------
    mahalanobis_calibration_ratio : Calibration ratio based on squared
        Mahalanobis distances.

    Notes
    -----
    Under correct calibration and Gaussian standardized returns,
    :math:`d^2 \sim \chi^2(n_{\text{features}})`, so
    :math:`P(d^2 > \chi^2_{\text{confidence\_level}}) =
    1 - \text{confidence\_level}`.
    """
    squared_distances = np.asarray(squared_distances)
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    valid = np.isfinite(squared_distances)
    if not valid.any():
        return float(np.nan)
    threshold = sst.chi2.ppf(confidence_level, df=n_features)
    return float(np.mean(squared_distances[valid] > threshold))


def qlike_loss(
    returns: ArrayLike,
    forecast_variance: ArrayLike,
) -> float:
    r"""QLIKE loss for univariate variance forecasts.

    .. math::

        \text{QLIKE} = \frac{1}{n} \sum_{t=1}^{n}
        \left( \log(\sigma_t^2) + \frac{r_t^2}{\sigma_t^2} \right)

    Lower values are better. For numerical stability, forecast variances are
    clipped below by a small positive constant before evaluating the score.

    In financial time series, QLIKE is often used as a comparative score when
    returns are heavy-tailed and realized variance is only an imperfect proxy
    for latent volatility.

    Parameters
    ----------
    returns : array-like of shape (n_observations,)
        Realized returns.

    forecast_variance : array-like of shape (n_observations,)
        Forecast variances for the same timestamps, expressed in squared
        return units.

    Returns
    -------
    float
        Mean QLIKE loss. Lower values are better; in expectation, the loss is minimized
        by the true conditional variance forecast.

    See Also
    --------
    portfolio_variance_qlike_loss : Multivariate QLIKE loss projected onto
        portfolio weights.
    """
    returns = np.asarray(returns)
    forecast_variance = np.asarray(forecast_variance)
    if returns.shape != forecast_variance.shape:
        raise ValueError("returns and forecast_variance must have the same shape.")
    variance = np.clip(forecast_variance, _NUMERICAL_THRESHOLD, None)
    return float(np.mean(np.log(variance) + (returns**2) / variance))


def _resolve_weights(
    active_covariance: FloatArray,
    portfolio_weights: ArrayLike | None,
    active_asset_indices: FloatArray,
) -> FloatArray:
    """Compute normalized weights on the active asset subset if portfolio weights are
    provided; otherwise, compute inverse-volatility weights.

    Parameters
    ----------
    active_covariance : ndarray of shape (n_active, n_active)
        Active covariance submatrix.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        User-provided portfolio weights on the original asset universe.
        If `None`, inverse-volatility weights are computed from
        `active_covariance`.

    active_asset_indices : ndarray of shape (n_active,)
        Indices of active assets for subsetting user weights.

    Returns
    -------
    w : ndarray of shape (n_portfolios, n_active)
        Row-normalized weight matrix on the active asset subset.
    """
    if portfolio_weights is not None:
        w = np.atleast_2d(np.asarray(portfolio_weights, dtype=np.float64))
        w = w[:, active_asset_indices]
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, _NUMERICAL_THRESHOLD)
        w = w / row_sums
    else:
        w = inverse_volatility_weights(active_covariance)[np.newaxis, :]
    return w


def _aggregated_return_and_effective_covariance(
    active_returns: FloatArray,
    active_covariance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute the aggregated return and effective (Hadamard-scaled) covariance.

    When all entries are finite this reduces to `(X.sum(0), h * cov)`.
    Otherwise, each covariance entry is scaled by the number of rows where both
    assets are observed, yielding the effective covariance :math:`H \odot \Sigma` used
    by the calibration metrics. This treats NaNs in `active_returns` as missing
    observations, while assets whose forecast covariance is marked inactive by
    NaN diagonal entries are handled upstream by subsetting.

    Parameters
    ----------
    active_returns : ndarray of shape (n_observations, n_active)
        Returns for active assets (may contain NaN).

    active_covariance : ndarray of shape (n_active, n_active)
        Covariance submatrix for active assets.

    Returns
    -------
    aggregated_return : ndarray of shape (n_active,)
        Aggregated return over the evaluation window, computed as the sum of
        finite observations per asset.

    effective_cov : ndarray of shape (n_active, n_active)
        Effective covariance :math:`H \odot \Sigma`.
    """
    active_mask = np.isfinite(active_returns)
    if active_mask.all():
        return (
            active_returns.sum(axis=0),
            active_returns.shape[0] * active_covariance,
        )
    active_returns_filled = np.where(active_mask, active_returns, 0.0)
    aggregated_return = active_returns_filled.sum(axis=0)
    active_mask_f = active_mask.astype(float)
    pairwise_obs_count = active_mask_f.T @ active_mask_f
    return aggregated_return, pairwise_obs_count * active_covariance


def _prepare_active_subset(
    covariance: FloatArray,
    X_test: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, int] | None:
    r"""Subset covariance and returns to jointly active assets.

    An asset is jointly active when its covariance diagonal is finite (available in the
    forecast) and it has at least one finite observation in `X_test`. In skfolio, NaN
    diagonal entries in the forecast covariance mark inactive assets, which are excluded
    from the evaluation. Assets with no finite observation in `X_test` (e.g.
    post-delisting periods) are also dropped and portfolio weights are renormalized to
    the remaining assets. NaN values in `X_test` are otherwise preserved so that callers
    can apply Hadamard scaling (zero-filled returns together with the pairwise
    observation count matrix :math:`H \odot \Sigma`).

    Parameters
    ----------
    covariance : ndarray of shape (n_assets, n_assets)
        Covariance matrix. In skfolio, NaN diagonal entries mark inactive assets.

    X_test : array-like of shape (n_observations, n_assets)
        Realized returns. NaN values are preserved (not zero-filled).

    Returns
    -------
    tuple of (
        ndarray of shape (n_active_assets, n_active_assets),
        ndarray of shape (n_observations, n_active_assets),
        ndarray of shape (n_active_assets,),
        int,
    ) or None
        Active covariance submatrix, corresponding return columns (may
        contain NaN), original column indices, and active asset count.
        Returns `None` when no assets are jointly active.
    """
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    active_mask = np.isfinite(np.diag(covariance))
    active_mask &= np.any(np.isfinite(X_test), axis=0)
    n_active_assets = int(active_mask.sum())
    if n_active_assets == 0:
        return None
    active_asset_indices = np.where(active_mask)[0]
    active_cov = covariance[np.ix_(active_asset_indices, active_asset_indices)]
    active_returns = X_test[:, active_asset_indices]
    return active_cov, active_returns, active_asset_indices, n_active_assets


def _get_covariance(estimator: skb.BaseEstimator) -> FloatArray:
    """Extract the covariance matrix from a covariance or prior estimator.

    Supports :class:`~skfolio.moments.BaseCovariance` (`estimator.covariance_`)
    and :class:`~skfolio.prior.BasePrior`
    (`estimator.return_distribution_.covariance`).

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted covariance estimator or prior estimator.

    Returns
    -------
    ndarray of shape (n_assets, n_assets)
        Covariance matrix extracted from the estimator.
    """
    if hasattr(estimator, "covariance_"):
        return estimator.covariance_
    if hasattr(estimator, "return_distribution_"):
        return estimator.return_distribution_.covariance
    raise AttributeError(
        f"{type(estimator).__name__} has neither `covariance_` nor "
        "`return_distribution_.covariance`."
    )
