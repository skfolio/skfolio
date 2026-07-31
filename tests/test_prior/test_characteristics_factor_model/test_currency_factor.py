"""Tests for currency factors in CharacteristicsFactorModel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skfolio._constants import _CURRENCY
from skfolio.factor_exposure import BaseFactorExposure
from skfolio.moments.variance import EWVariance
from skfolio.prior import CharacteristicsFactorModel

from .conftest import make_panel, passthrough_factor


class StaticCurrencyFactor(BaseFactorExposure, stateless=True):
    """One-hot currency exposures from fixed asset currency assignments."""

    def __init__(self, currency_names, currency_codes) -> None:
        super().__init__(family=_CURRENCY)
        self.currency_names = currency_names
        self.currency_codes = currency_codes

    def fit_transform(self, X, y=None, **fit_params):
        currency_names = np.asarray(self.currency_names)
        currency_codes = np.asarray(self.currency_codes, dtype=int)
        exposures = np.zeros(
            (X.n_observations, X.n_assets, len(currency_names)), dtype=float
        )
        exposures[:, np.arange(X.n_assets), currency_codes] = 1.0
        self.factor_names_ = currency_names
        return exposures


def _make_currency_data(n_obs=80, n_assets=6):
    rng = np.random.default_rng(42)
    beta = np.linspace(0.6, 1.4, n_assets)
    local_factor_returns = rng.normal(0.0, 0.01, size=n_obs)
    local_returns = local_factor_returns[:, None] * beta[None, :]
    beta_field = np.broadcast_to(beta, local_returns.shape).copy()

    panel, X = make_panel(local_returns, extra_fields={"beta": beta_field})

    currency_names = np.array(["USD", "EUR"])
    currency_codes = np.array([0, 0, 0, 1, 1, 1])
    currency_returns = pd.DataFrame(
        {
            "USD": rng.normal(0.0, 0.002, size=n_obs),
            "EUR": rng.normal(0.0, 0.003, size=n_obs),
        },
        index=panel.observations,
    )
    currency_factor = StaticCurrencyFactor(currency_names, currency_codes)
    return panel, X, local_factor_returns, currency_returns, currency_factor


def _make_model(currency_factor):
    return CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        currency_factor=currency_factor,
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
        min_regression_assets=3,
        idio_variance_estimator=EWVariance(half_life=10, min_observations=1),
    )


def test_currency_factors_are_stored_as_direct_factors():
    panel, X, local_factor_returns, currency_returns, currency_factor = (
        _make_currency_data()
    )
    model = _make_model(currency_factor)
    model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    fm = model.factor_model_
    assert list(fm.factor_names) == ["beta", "USD", "EUR"]
    assert list(fm.factor_families) == ["market", _CURRENCY, _CURRENCY]
    np.testing.assert_allclose(fm.factor_returns[:, 0], local_factor_returns[1:])
    np.testing.assert_allclose(
        fm.factor_returns[:, 1:], currency_returns.iloc[1:].to_numpy()
    )


def test_currency_decomposition_matches_return_distribution():
    panel, X, _local_factor_returns, currency_returns, currency_factor = (
        _make_currency_data()
    )
    model = _make_model(currency_factor)
    model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    fm = model.factor_model_
    expected_mu = fm.loading_matrix @ fm.factor_mu + fm.idio_mu
    expected_covariance = fm.loading_matrix @ fm.factor_covariance @ fm.loading_matrix.T
    if fm.idio_covariance.ndim == 1:
        expected_covariance[np.diag_indices_from(expected_covariance)] += (
            fm.idio_covariance
        )
    else:
        expected_covariance += fm.idio_covariance

    np.testing.assert_allclose(model.return_distribution_.mu, expected_mu)
    np.testing.assert_allclose(
        model.return_distribution_.covariance, expected_covariance
    )


def test_currency_factor_with_constrained_families():
    rng = np.random.default_rng(7)
    n_obs, n_assets = 100, 6

    f_mkt = rng.normal(0.0, 0.01, size=n_obs)
    f_ind = rng.normal(0.0, 0.006, size=(n_obs, 2))
    eps = rng.normal(0.0, 0.001, size=(n_obs, n_assets))

    ind_1 = np.zeros((n_obs, n_assets))
    ind_2 = np.zeros((n_obs, n_assets))
    ind_1[:, :3] = 1.0
    ind_2[:, 3:] = 1.0

    returns = f_mkt[:, None] + ind_1 * f_ind[:, [0]] + ind_2 * f_ind[:, [1]] + eps
    market = np.ones((n_obs, n_assets))
    market_cap = np.ones((n_obs, n_assets))
    market_cap[:, :3] = 2.0

    panel, X = make_panel(
        returns,
        extra_fields={
            "market": market,
            "ind_1": ind_1,
            "ind_2": ind_2,
        },
        market_cap=market_cap,
    )
    currency_returns = pd.DataFrame(
        {
            "USD": rng.normal(0.0, 0.002, size=n_obs),
            "EUR": rng.normal(0.0, 0.003, size=n_obs),
        },
        index=panel.observations,
    )
    currency_factor = StaticCurrencyFactor(
        currency_names=np.array(["USD", "EUR"]),
        currency_codes=np.array([0, 1, 0, 1, 0, 1]),
    )

    model = CharacteristicsFactorModel(
        factors=[
            ("market", passthrough_factor("market", family="market")),
            ("ind_1", passthrough_factor("ind_1", family="industry")),
            ("ind_2", passthrough_factor("ind_2", family="industry")),
        ],
        currency_factor=currency_factor,
        constrained_families=[("industry", None)],
        benchmark_mcap_power=1,
        regression_mcap_power=1,
        min_regression_assets=4,
        idio_variance_estimator=EWVariance(half_life=10, min_observations=1),
    )
    model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    fm = model.factor_model_
    assert list(fm.factor_names) == ["market", "ind_1", "ind_2", "USD", "EUR"]
    basis = fm.family_constraint_basis
    assert basis is not None
    assert basis.n_full_factors == len(fm.factor_names)
    np.testing.assert_array_equal(
        basis.reduced_factor_names(fm.factor_names),
        ["market", "ind_2", "USD", "EUR"],
    )
    np.testing.assert_allclose(
        fm.factor_returns[:, -2:], currency_returns.iloc[1:].to_numpy()
    )

    ind_weights = np.array([6.0 / 9.0, 3.0 / 9.0])
    ind_returns = fm.factor_returns[:, [1, 2]]
    np.testing.assert_allclose(ind_returns @ ind_weights, 0.0, atol=1e-12)

    expected_covariance = fm.loading_matrix @ fm.factor_covariance @ fm.loading_matrix.T
    if fm.idio_covariance.ndim == 1:
        expected_covariance[np.diag_indices_from(expected_covariance)] += (
            fm.idio_covariance
        )
    else:
        expected_covariance += fm.idio_covariance
    np.testing.assert_allclose(
        model.return_distribution_.covariance, expected_covariance
    )


def test_regression_diagnostics_exclude_currency_factors():
    panel, X, _local_factor_returns, currency_returns, currency_factor = (
        _make_currency_data()
    )
    model = _make_model(currency_factor)
    model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    fm = model.factor_model_
    assert list(fm.cs_regression_t_stats.columns) == ["beta"]
    assert list(fm.exposure_vif.columns) == ["beta"]
    assert np.isfinite(fm.cs_regression_scores["r2"]).all()


def test_realized_attribution_uncertainty_marks_currency_as_direct():
    panel, X, _local_factor_returns, currency_returns, currency_factor = (
        _make_currency_data()
    )
    model = _make_model(currency_factor)
    model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    fm = model.factor_model_
    weights = np.full(fm.loading_matrix.shape[0], 1 / fm.loading_matrix.shape[0])
    lagged_exposures, factor_returns, idio_returns = fm._aligned(
        ["exposures", "factor_returns", "idio_returns"]
    )
    asset_returns = (lagged_exposures @ factor_returns[:, :, np.newaxis]).squeeze(
        -1
    ) + idio_returns
    portfolio_returns = np.zeros(len(fm.factor_returns))
    portfolio_returns[fm.exposure_lag :] = asset_returns @ weights

    attribution = fm.realized_attribution(
        weights=weights,
        portfolio_returns=portfolio_returns,
        compute_asset_breakdowns=False,
        compute_uncertainty=True,
    )

    currency_mask = fm.factor_families == _CURRENCY
    assert np.isnan(attribution.factors.mu_contrib_uncertainty[currency_mask]).all()
    assert np.all(
        np.isfinite(attribution.factors.mu_contrib_uncertainty[~currency_mask])
    )


def test_partial_fit_matches_fit_with_currency_factor():
    panel, X, _local_factor_returns, currency_returns, currency_factor = (
        _make_currency_data(n_obs=100)
    )
    split = 45

    model_full = _make_model(currency_factor)
    model_full.fit(X, characteristics=panel, currency_excess_returns=currency_returns)

    model_pf = _make_model(currency_factor)
    model_pf.partial_fit(
        X.iloc[:split],
        characteristics=panel[:split],
        currency_excess_returns=currency_returns.iloc[:split],
    )
    model_pf.partial_fit(
        X.iloc[split:],
        characteristics=panel[split:],
        currency_excess_returns=currency_returns.iloc[split:],
    )

    np.testing.assert_array_equal(
        model_pf.factor_model_.factor_names, model_full.factor_model_.factor_names
    )
    np.testing.assert_allclose(
        model_pf.factor_model_.factor_returns,
        model_full.factor_model_.factor_returns,
    )
    np.testing.assert_allclose(
        model_pf.factor_model_.exposures,
        model_full.factor_model_.exposures,
    )
    np.testing.assert_allclose(
        model_pf.return_distribution_.mu, model_full.return_distribution_.mu
    )
    np.testing.assert_allclose(
        model_pf.return_distribution_.covariance,
        model_full.return_distribution_.covariance,
    )


def test_currency_returns_required_with_currency_factor():
    panel, X, _local_factor_returns, _currency_returns, currency_factor = (
        _make_currency_data()
    )
    model = _make_model(currency_factor)

    with np.testing.assert_raises_regex(ValueError, "currency_excess_returns"):
        model.fit(X, characteristics=panel)


def test_currency_returns_rejected_without_currency_factor():
    panel, X, _local_factor_returns, currency_returns, _currency_factor = (
        _make_currency_data()
    )
    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        min_regression_assets=3,
    )

    with np.testing.assert_raises_regex(ValueError, "currency_excess_returns"):
        model.fit(X, characteristics=panel, currency_excess_returns=currency_returns)


def test_currency_family_is_reserved_for_currency_factor():
    panel, X, _local_factor_returns, _currency_returns, _currency_factor = (
        _make_currency_data()
    )
    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family=_CURRENCY))],
        min_regression_assets=3,
    )

    with np.testing.assert_raises_regex(ValueError, "reserved"):
        model.fit(X, characteristics=panel)


def test_currency_family_cannot_be_constrained():
    panel, X, _local_factor_returns, _currency_returns, _currency_factor = (
        _make_currency_data()
    )
    model = CharacteristicsFactorModel(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        constrained_families=[(_CURRENCY, None)],
        min_regression_assets=3,
    )

    with np.testing.assert_raises_regex(ValueError, "constrained_families"):
        model.fit(X, characteristics=panel)
