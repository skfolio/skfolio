"""Test that inverse-variance regression weights handle warm-up NaN gracefully.

The variance estimator returns NaN for assets that are active but have not yet
accumulated `min_observations` valid observations (warm-up period).  Before
the fix, NaN would propagate through normalization and blending into
`regression_weights`, corrupting the second-pass WLS regression. Missing
per-asset inverse-variance weights are set to zero in the inverse-variance
component; rows with no ready variance estimates fall back to cap weights.
"""

from __future__ import annotations

import numpy as np

from skfolio.moments import EWCovariance
from skfolio.moments.variance import EWVariance
from skfolio.prior import CharacteristicsFactorModel

from .conftest import make_panel, passthrough_factor


def test_no_nan_regression_weights_with_late_listing():
    """Regression weights must be NaN-free when active assets are in warm-up."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 100, 10
    late_asset = 5
    listing_obs = 20

    betas = rng.uniform(0.5, 1.5, size=n_assets)
    f_true = rng.normal(0, 0.01, size=n_obs)
    eps = rng.normal(0, 0.005, size=(n_obs, n_assets))
    returns = betas[None, :] * f_true[:, None] + eps

    returns[:listing_obs, late_asset] = np.nan

    active_mask = np.ones((n_obs, n_assets), dtype=bool)
    active_mask[:listing_obs, late_asset] = False

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    betas_2d[:listing_obs, late_asset] = np.nan

    market_cap = np.ones((n_obs, n_assets))
    market_cap[:listing_obs, late_asset] = np.nan

    panel, X = make_panel(
        returns,
        extra_fields={"beta": betas_2d},
        market_cap=market_cap,
        active_mask=active_mask,
    )

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
        min_regression_assets=3,
        inv_idio_variance_weight_shrinkage=0.5,
        idio_variance_estimator=EWVariance(half_life=10, min_observations=10),
    )
    model.fit(X, characteristics=panel)

    rw = model.factor_model_.regression_weights
    assert not np.any(np.isnan(rw)), (
        f"regression_weights has {np.isnan(rw).sum()} NaN entries"
    )


def test_inv_var_weight_reacts_with_one_bar_delay():
    """A date-t idiosyncratic shock must affect date t + 1 weights."""
    rng = np.random.default_rng(123)
    n_obs, n_assets = 80, 24
    shock_obs = 42

    betas = np.linspace(0.7, 1.3, n_assets)
    factor_returns = rng.normal(0, 0.01, size=n_obs)
    eps = rng.normal(0, 0.0015, size=(n_obs, n_assets))
    eps[shock_obs, 0] = 0.15
    returns = betas[None, :] * factor_returns[:, None] + eps

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    panel, X = make_panel(returns, extra_fields={"beta": betas_2d})

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
        min_regression_assets=12,
        inv_idio_variance_weight_shrinkage=1.0,
        idio_variance_estimator=EWVariance(half_life=2, min_observations=2),
    )
    model.fit(X, characteristics=panel)

    rw = model.factor_model_.regression_weights
    shock_row = shock_obs - model.factor_model_.exposure_lag

    before = rw[shock_row - 1, 0]
    at_shock = rw[shock_row, 0]
    after = rw[shock_row + 1, 0]

    assert np.isfinite([before, at_shock, after]).all()
    assert 0.5 * before < at_shock < 1.5 * before
    assert after < 0.35 * at_shock


def test_first_observation_falls_back_to_cap_weights():
    """The first GLS row has no prior variance and uses cap weights."""
    rng = np.random.default_rng(321)
    n_obs, n_assets = 120, 8

    betas = rng.uniform(0.5, 1.5, size=n_assets)
    factor_returns = rng.normal(0, 0.01, size=n_obs)
    returns = betas[None, :] * factor_returns[:, None]
    returns += rng.normal(0, 0.003, size=(n_obs, n_assets))

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    market_cap = np.broadcast_to(
        np.geomspace(1.0, 5.0, n_assets), (n_obs, n_assets)
    ).copy()
    panel, X = make_panel(
        returns,
        extra_fields={"beta": betas_2d},
        market_cap=market_cap,
    )

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=1,
        min_regression_assets=6,
        inv_idio_variance_weight_shrinkage=1.0,
        idio_variance_estimator=EWVariance(half_life=5, min_observations=5),
    )
    model.fit(X, characteristics=panel)

    expected = market_cap[0] / market_cap[0].sum()
    np.testing.assert_allclose(
        model.factor_model_.regression_weights[0],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_blend_matches_nominal_shrinkage():
    """The inverse-variance component normalizes on eligible assets only."""
    rng = np.random.default_rng(456)
    n_obs, n_assets = 60, 14
    missing_obs = 35
    missing_assets = np.array([1, 5, 11])
    shrinkage = 0.6

    betas = rng.uniform(0.6, 1.4, size=n_assets)
    factor_returns = rng.normal(0, 0.01, size=n_obs)
    returns = betas[None, :] * factor_returns[:, None]
    returns += rng.normal(0, 0.004, size=(n_obs, n_assets))
    returns[missing_obs, missing_assets] = np.nan

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    market_cap = np.broadcast_to(
        np.geomspace(1.0, 4.0, n_assets), (n_obs, n_assets)
    ).copy()
    panel, X = make_panel(
        returns,
        extra_fields={"beta": betas_2d},
        market_cap=market_cap,
    )

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=1,
        min_regression_assets=10,
        inv_idio_variance_weight_shrinkage=shrinkage,
        idio_variance_estimator=EWVariance(half_life=4, min_observations=2),
    )
    model.fit(X, characteristics=panel)

    lag = model.factor_model_.exposure_lag
    row = missing_obs - lag
    eligible = np.isfinite(returns[missing_obs])
    cap_component = market_cap[missing_obs - lag, eligible]
    cap_component = cap_component / cap_component.sum()

    weights = model.factor_model_.regression_weights[row]
    inv_var_component = (
        weights[eligible] - (1 - shrinkage) * cap_component
    ) / shrinkage

    np.testing.assert_allclose(inv_var_component.sum(), 1.0, rtol=1e-12, atol=1e-12)
    assert np.all(inv_var_component >= -1e-12)
    np.testing.assert_allclose(weights[~eligible], 0.0, atol=1e-12)


def test_idio_corr_threshold_with_partial_warmup():
    """Correlation overlay keeps NaN rows for assets still warming up."""
    rng = np.random.default_rng(789)
    n_obs, n_assets = 90, 10
    listing_obs = 84
    late_assets = np.array([8, 9])

    betas = rng.uniform(0.7, 1.3, size=n_assets)
    factor_returns = rng.normal(0, 0.01, size=n_obs)
    returns = betas[None, :] * factor_returns[:, None]
    returns += rng.normal(0, 0.003, size=(n_obs, n_assets))
    returns[:listing_obs, late_assets] = np.nan

    active_mask = np.ones((n_obs, n_assets), dtype=bool)
    active_mask[:listing_obs, late_assets] = False

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    betas_2d[:listing_obs, late_assets] = np.nan

    market_cap = np.ones((n_obs, n_assets))
    market_cap[:listing_obs, late_assets] = np.nan

    panel, X = make_panel(
        returns,
        extra_fields={"beta": betas_2d},
        market_cap=market_cap,
        active_mask=active_mask,
    )

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
        min_regression_assets=6,
        idio_variance_estimator=EWVariance(half_life=3, min_observations=10),
        idio_corr_threshold=0.1,
    )
    model.fit(X, characteristics=panel)

    fm = model.factor_model_
    cov = fm.idio_covariance
    latest_var = fm.idio_variances[-1]
    warmup = np.isnan(latest_var)

    np.testing.assert_array_equal(warmup, np.isin(np.arange(n_assets), late_assets))
    assert cov.ndim == 2
    assert np.isnan(cov[warmup]).all()
    assert np.isnan(cov[:, warmup]).all()

    ready = ~warmup
    sub_block = cov[np.ix_(ready, ready)]
    np.linalg.cholesky(sub_block)
    np.testing.assert_allclose(np.diag(sub_block), latest_var[ready], rtol=1e-10)


def test_idio_corr_nan_correlations_treated_as_zero():
    """Correlation warmup degrades to a diagonal idio covariance."""
    rng = np.random.default_rng(987)
    n_obs, n_assets = 50, 9

    betas = rng.uniform(0.7, 1.3, size=n_assets)
    factor_returns = rng.normal(0, 0.01, size=n_obs)
    returns = betas[None, :] * factor_returns[:, None]
    returns += rng.normal(0, 0.003, size=(n_obs, n_assets))

    betas_2d = np.broadcast_to(betas, (n_obs, n_assets)).copy()
    panel, X = make_panel(returns, extra_fields={"beta": betas_2d})

    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
        min_regression_assets=6,
        idio_variance_estimator=EWVariance(half_life=2, min_observations=2),
        idio_corr_estimator=EWCovariance(half_life=20, min_observations=200),
        idio_corr_threshold=0.1,
    )
    model.fit(X, characteristics=panel)

    cov = model.factor_model_.idio_covariance
    latest_var = model.factor_model_.idio_variances[-1]
    off_diag = cov.copy()
    np.fill_diagonal(off_diag, 0.0)

    np.testing.assert_allclose(np.diag(cov), latest_var, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)
