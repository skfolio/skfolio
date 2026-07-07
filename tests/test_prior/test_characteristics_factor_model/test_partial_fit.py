"""Tests for CharacteristicsFactorModel.partial_fit equivalence.

Verifies that calling ``partial_fit`` in batches produces the same output as a
single ``fit`` call over the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.moments.variance import EWVariance
from skfolio.prior import CharacteristicsFactorModel

from .conftest import make_panel, passthrough_factor


def _make_single_factor_data(n_obs, n_assets, seed=42):
    """Generate a single-factor DGP with constant betas."""
    rng = np.random.default_rng(seed)
    beta_true = rng.uniform(0.5, 1.5, size=n_assets)
    sigma_f = 0.01
    sigma_eps = rng.uniform(0.005, 0.02, size=n_assets)

    f_true = rng.normal(0, sigma_f, size=n_obs)
    eps = rng.normal(0, 1, size=(n_obs, n_assets)) * sigma_eps
    returns = beta_true[None, :] * f_true[:, None] + eps

    betas_field = np.broadcast_to(beta_true, (n_obs, n_assets)).copy()
    panel, X = make_panel(returns, extra_fields={"beta": betas_field})
    return panel, X, beta_true, sigma_f, sigma_eps


def _make_industry_data(n_obs, n_ind=3, assets_per_ind=10, seed=888, drift_caps=False):
    """Generate a market + industry DGP with heterogeneous cap weights.

    With `drift_caps=True`, market caps drift linearly over time so the
    family-constraint ratios are time-varying.
    """
    rng = np.random.default_rng(seed)
    n_assets = n_ind * assets_per_ind
    ind_mcaps = [3.0, 2.0, 1.0]

    f_mkt = rng.normal(0, 0.01, size=n_obs)
    f_ind = rng.normal(0, 0.01, size=(n_obs, n_ind))
    eps = rng.normal(0, 0.005, size=(n_obs, n_assets))

    ind_dummies = np.zeros((n_obs, n_assets, n_ind))
    mcap = np.ones((n_obs, n_assets))
    for k in range(n_ind):
        s, e = k * assets_per_ind, (k + 1) * assets_per_ind
        ind_dummies[:, s:e, k] = 1.0
        mcap[:, s:e] = ind_mcaps[k]
    if drift_caps:
        # Linear drift with industry-dependent slope so the constraint ratios
        # change at every observation.
        t_grid = np.linspace(0, 1, n_obs)[:, None]
        slopes = np.tile(
            np.linspace(-0.5, 0.5, n_ind).repeat(assets_per_ind), (n_obs, 1)
        )
        mcap = mcap * (1.0 + slopes * t_grid)

    returns = f_mkt[:, None] + np.einsum("tnk,tk->tn", ind_dummies, f_ind) + eps

    mkt_exp = np.ones((n_obs, n_assets))
    extra_fields = {
        "mkt_exp": mkt_exp,
        "ind_1": ind_dummies[:, :, 0],
        "ind_2": ind_dummies[:, :, 1],
        "ind_3": ind_dummies[:, :, 2],
    }
    panel, X = make_panel(returns, extra_fields=extra_fields, market_cap=mcap)
    return panel, X


def _make_model(**overrides):
    """Build a CharacteristicsFactorModel with reproducible sub-estimators."""
    defaults = dict(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
    )
    defaults.update(overrides)
    return CharacteristicsFactorModel(**defaults)


def _make_industry_model(**overrides):
    """Build a model with market + 3 industry factors and basket-neutral."""
    defaults = dict(
        factors=[
            ("market", passthrough_factor("mkt_exp", family="market")),
            ("ind_1", passthrough_factor("ind_1", family="industry")),
            ("ind_2", passthrough_factor("ind_2", family="industry")),
            ("ind_3", passthrough_factor("ind_3", family="industry")),
        ],
        constrained_families=[("industry", None)],
        benchmark_mcap_power=1,
        regression_mcap_power=1,
    )
    defaults.update(overrides)
    return CharacteristicsFactorModel(**defaults)


def _assert_history_arrays_row_aligned(model):
    fm = model.factor_model_
    expected = len(fm.observations)
    arrays = {
        "factor_returns": fm.factor_returns,
        "idio_returns": fm.idio_returns,
        "idio_variances": fm.idio_variances,
        "exposures": fm.exposures,
        "regression_weights": fm.regression_weights,
        "benchmark_weights": fm.benchmark_weights,
    }
    for name, values in arrays.items():
        assert values.shape[0] == expected, (
            f"{name} has {values.shape[0]} rows, expected {expected}"
        )


def test_first_partial_fit_requires_enough_observations():
    panel, X, _beta_true, _sigma_f, _sigma_eps = _make_single_factor_data(
        n_obs=5,
        n_assets=50,
    )

    model = _make_model()

    with pytest.raises(
        ValueError,
        match="first `partial_fit` call must contain enough data",
    ):
        model.partial_fit(X.iloc[:1], characteristics=panel[:1])


class TestPartialFitEquivalence:
    """fit(all) must equal partial_fit(part1) + partial_fit(part2)."""

    N_OBS = 200
    N_ASSETS = 50
    SEED = 42

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, _beta_true, _sigma_f, _sigma_eps = _make_single_factor_data(
            cls.N_OBS, cls.N_ASSETS, cls.SEED
        )

        # Full fit
        model_full = _make_model()
        model_full.fit(X, characteristics=panel)

        # Split partial_fit at midpoint
        split = cls.N_OBS // 2
        panel1, X1 = panel[:split], X.iloc[:split]
        panel2, X2 = panel[split:], X.iloc[split:]

        model_pf = _make_model()
        model_pf.partial_fit(X1, characteristics=panel1)
        model_pf.partial_fit(X2, characteristics=panel2)

        cls.model_full = model_full
        cls.model_pf = model_pf
        cls.split = split

    # -- Moment estimates (must be numerically identical) --

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_regression_weights_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.regression_weights,
            self.model_full.factor_model_.regression_weights,
            rtol=1e-10,
        )


class TestPartialFitInverseVarianceShrinkage:
    """`partial_fit` must match `fit` on the feasible GLS weighting path."""

    N_OBS = 240
    N_ASSETS = 40
    SEED = 909

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        kwargs = dict(
            inv_idio_variance_weight_shrinkage=0.5,
            idio_variance_estimator=EWVariance(half_life=8, min_observations=2),
        )
        model_full = _make_model(**kwargs)
        model_full.fit(X, characteristics=panel)

        model_pf = _make_model(**kwargs)
        for start, stop in zip(
            [0, 57, 121, 200], [57, 121, 200, cls.N_OBS], strict=True
        ):
            model_pf.partial_fit(X.iloc[start:stop], characteristics=panel[start:stop])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_regression_weights_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.regression_weights,
            self.model_full.factor_model_.regression_weights,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_asset_mu(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.mu,
            self.model_full.return_distribution_.mu,
            rtol=1e-10,
        )

    def test_covariance_sqrt(self):
        sqrt_pf = self.model_pf.return_distribution_.covariance_sqrt
        sqrt_full = self.model_full.return_distribution_.covariance_sqrt
        for b_pf, b_full in zip(sqrt_pf.components, sqrt_full.components, strict=False):
            np.testing.assert_allclose(b_pf, b_full, rtol=1e-10)
        if sqrt_pf.diagonal is not None:
            np.testing.assert_allclose(sqrt_pf.diagonal, sqrt_full.diagonal, rtol=1e-10)

    def test_systematic_returns(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.returns,
            self.model_full.return_distribution_.returns,
            rtol=1e-10,
        )

    # -- Factor model outputs --

    def test_factor_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_covariance,
            self.model_full.factor_model_.factor_covariance,
            rtol=1e-10,
        )

    def test_factor_mu(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_mu,
            self.model_full.factor_model_.factor_mu,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.loading_matrix,
            self.model_full.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_idio_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_covariance,
            self.model_full.factor_model_.idio_covariance,
            rtol=1e-10,
        )

    def test_idio_mu(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_mu,
            self.model_full.factor_model_.idio_mu,
            rtol=1e-10,
        )

    # -- Accumulated time series (shapes and values) --

    def test_observations_accumulated(self):
        np.testing.assert_array_equal(
            self.model_pf.factor_model_.observations,
            self.model_full.factor_model_.observations,
        )

    def test_factor_returns_accumulated2(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_idio_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_returns,
            self.model_full.factor_model_.idio_returns,
            rtol=1e-10,
        )

    def test_idio_variances_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_variances,
            self.model_full.factor_model_.idio_variances,
            rtol=1e-10,
        )

    def test_exposures_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.exposures,
            self.model_full.factor_model_.exposures,
            rtol=1e-10,
        )

    def test_regression_weights_accumulated2(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.regression_weights,
            self.model_full.factor_model_.regression_weights,
            rtol=1e-10,
        )


class TestPartialFitThreeWaySplit:
    """Verify partial_fit with three unequal batches."""

    N_OBS = 300
    N_ASSETS = 50
    SEED = 99

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_full = _make_model()
        model_full.fit(X, characteristics=panel)

        s1, s2 = 80, 200
        model_pf = _make_model()
        model_pf.partial_fit(X.iloc[:s1], characteristics=panel[:s1])
        model_pf.partial_fit(X.iloc[s1:s2], characteristics=panel[s1:s2])
        model_pf.partial_fit(X.iloc[s2:], characteristics=panel[s2:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_asset_mu(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.mu,
            self.model_full.return_distribution_.mu,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_observations_accumulated(self):
        np.testing.assert_array_equal(
            self.model_pf.factor_model_.observations,
            self.model_full.factor_model_.observations,
        )

    def test_idio_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_returns,
            self.model_full.factor_model_.idio_returns,
            rtol=1e-10,
        )

    def test_idio_variances_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_variances,
            self.model_full.factor_model_.idio_variances,
            rtol=1e-10,
        )


class TestPartialFitExposureLagShortBatches:
    """partial_fit must preserve lag alignment when later batches are short."""

    N_OBS = 240
    N_ASSETS = 50
    SEED = 123
    EXPOSURE_LAG = 3

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_full = _make_model(exposure_lag=cls.EXPOSURE_LAG)
        model_full.fit(X, characteristics=panel)

        cuts = [80, 81, 83, 140]
        starts = [0, *cuts]
        stops = [*cuts, cls.N_OBS]

        model_pf = _make_model(exposure_lag=cls.EXPOSURE_LAG)
        for start, stop in zip(starts, stops, strict=True):
            model_pf.partial_fit(
                X.iloc[start:stop],
                characteristics=panel[start:stop],
            )

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_observations_accumulated(self):
        np.testing.assert_array_equal(
            self.model_pf.factor_model_.observations,
            self.model_full.factor_model_.observations,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_idio_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_returns,
            self.model_full.factor_model_.idio_returns,
            rtol=1e-10,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )


class TestPartialFitWithScores:
    """Verify that regression scores match between partial_fit and full fit.

    Scores come from :meth:`~skfolio.prior.FactorModel.cs_regression_scores` and
    :meth:`~skfolio.prior.FactorModel.cs_regression_t_stats`; they match when the
    underlying stored history matches.
    """

    N_OBS = 200
    N_ASSETS = 50
    SEED = 77

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_full = _make_model()
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_model()
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_r2_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.cs_regression_scores["r2"].values,
            self.model_full.factor_model_.cs_regression_scores["r2"].values,
            rtol=1e-10,
        )

    def test_t_stats_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.cs_regression_t_stats.values,
            self.model_full.factor_model_.cs_regression_t_stats.values,
            rtol=1e-10,
        )


class TestPartialFitMaxHistory:
    """Verify that max_history truncates accumulated time series."""

    N_OBS = 300
    N_ASSETS = 50
    SEED = 55
    MAX_HISTORY = 120

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model = _make_model(max_history=cls.MAX_HISTORY)
        s1, s2 = 100, 200
        model.partial_fit(X.iloc[:s1], characteristics=panel[:s1])
        model.partial_fit(X.iloc[s1:s2], characteristics=panel[s1:s2])
        model.partial_fit(X.iloc[s2:], characteristics=panel[s2:])

        # Reference: full fit with same max_history (single batch, no
        # accumulation needed -- but truncation still applies)
        model_full = _make_model(max_history=cls.MAX_HISTORY)
        model_full.fit(X, characteristics=panel)

        cls.model = model
        cls.model_full = model_full

    def test_factor_returns_capped(self):
        assert self.model.factor_model_.factor_returns.shape[0] == self.MAX_HISTORY

    def test_idio_returns_capped(self):
        assert self.model.factor_model_.idio_returns.shape[0] == self.MAX_HISTORY

    def test_idio_variances_capped(self):
        assert self.model.factor_model_.idio_variances.shape[0] == self.MAX_HISTORY

    def test_exposures_capped(self):
        assert self.model.factor_model_.exposures.shape[0] == self.MAX_HISTORY

    def test_observations_capped(self):
        assert len(self.model.factor_model_.observations) == self.MAX_HISTORY

    def test_systematic_returns_capped(self):
        assert self.model.return_distribution_.returns.shape[0] == self.MAX_HISTORY

    def test_keeps_most_recent(self):
        """Truncated time series must contain the tail of the full history."""
        np.testing.assert_allclose(
            self.model.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_moments_match_full_fit(self):
        """Moments must be identical -- max_history only affects stored history."""
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            self.model.return_distribution_.mu,
            self.model_full.return_distribution_.mu,
            rtol=1e-10,
        )


class TestPartialFitMaxHistorySnapshot:
    """Earlier `FactorModel` snapshots must not mutate after truncation."""

    N_OBS = 260
    N_ASSETS = 40
    SEED = 56
    MAX_HISTORY = 60

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model = _make_model(max_history=cls.MAX_HISTORY)
        model.partial_fit(X.iloc[:110], characteristics=panel[:110])
        model.partial_fit(X.iloc[110:170], characteristics=panel[110:170])

        snapshot = model.factor_model_
        factor_returns_copy = snapshot.factor_returns.copy()
        idio_returns_copy = snapshot.idio_returns.copy()

        model.partial_fit(X.iloc[170:220], characteristics=panel[170:220])
        model.partial_fit(X.iloc[220:], characteristics=panel[220:])

        cls.snapshot = snapshot
        cls.factor_returns_copy = factor_returns_copy
        cls.idio_returns_copy = idio_returns_copy

    def test_factor_returns_snapshot_immutable(self):
        np.testing.assert_allclose(
            self.snapshot.factor_returns,
            self.factor_returns_copy,
            rtol=0,
            atol=0,
        )

    def test_idio_returns_snapshot_immutable(self):
        np.testing.assert_allclose(
            self.snapshot.idio_returns,
            self.idio_returns_copy,
            rtol=0,
            atol=0,
        )


class TestHistoryArraysRowAligned:
    """Stored time-series arrays must share the observation axis length."""

    def test_fit_history_arrays_row_aligned(self):
        panel, X, *_ = _make_single_factor_data(n_obs=140, n_assets=35, seed=71)

        model = _make_model()
        model.fit(X, characteristics=panel)

        _assert_history_arrays_row_aligned(model)

    def test_partial_fit_history_arrays_row_aligned(self):
        panel, X, *_ = _make_single_factor_data(n_obs=180, n_assets=35, seed=72)

        model = _make_model()
        for start, stop in zip([0, 53, 101], [53, 101, 180], strict=True):
            model.partial_fit(X.iloc[start:stop], characteristics=panel[start:stop])

        _assert_history_arrays_row_aligned(model)


class TestPartialFitMaxHistoryScores:
    """Verify max_history truncates FactorModel data used by regression scores."""

    N_OBS = 300
    N_ASSETS = 50
    SEED = 66
    MAX_HISTORY = 100

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model = _make_model(max_history=cls.MAX_HISTORY)
        model.partial_fit(X.iloc[:150], characteristics=panel[:150])
        model.partial_fit(X.iloc[150:], characteristics=panel[150:])

        cls.model = model

    def test_r2_capped(self):
        r2 = self.model.factor_model_.cs_regression_scores["r2"]
        assert len(r2) <= self.MAX_HISTORY

    def test_t_stats_capped(self):
        t = self.model.factor_model_.cs_regression_t_stats
        assert t.shape[0] <= self.MAX_HISTORY


class TestPartialFitBasketNeutral:
    """partial_fit with constrained_families must match fit(all).

    Uses market + 3 industry factors with ``constrained_families``.
    The resolved factor_to_drop must be cached on the first call so subsequent
    batches use the same basis.
    """

    N_OBS = 500
    SEED = 888

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X = _make_industry_data(cls.N_OBS, seed=cls.SEED)

        model_full = _make_industry_model()
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_industry_model()
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_factor_names_match(self):
        assert list(self.model_pf.factor_model_.factor_names) == list(
            self.model_full.factor_model_.factor_names
        )

    def test_factor_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_covariance,
            self.model_full.factor_model_.factor_covariance,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.loading_matrix,
            self.model_full.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_observations_accumulated(self):
        np.testing.assert_array_equal(
            self.model_pf.factor_model_.observations,
            self.model_full.factor_model_.observations,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_resolved_constraints_cached(self):
        """After partial_fit the resolved constraints must not contain None."""
        for (
            _family,
            factor_to_drop,
        ) in self.model_pf._constrained_families:
            assert factor_to_drop is not None


class TestPartialFitBasketNeutralDriftingCaps:
    """partial_fit with constrained_families and time-varying caps must match
    fit(all).

    Drifting market caps make the family-constraint ratios change at every
    observation, exercising the constraint-ratio buffer that carries the
    formation-date ratios across batch boundaries.
    """

    N_OBS = 500
    SEED = 888

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X = _make_industry_data(cls.N_OBS, seed=cls.SEED, drift_caps=True)

        model_full = _make_industry_model()
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_industry_model()
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_factor_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_covariance,
            self.model_full.factor_model_.factor_covariance,
            rtol=1e-10,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )


class TestPartialFitBasketNeutralBasisMode:
    """partial_fit with basket-neutral constraints must match fit(all)."""

    N_OBS = 500
    SEED = 888

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X = _make_industry_data(cls.N_OBS, seed=cls.SEED)

        model_full = _make_industry_model()
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_industry_model()
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_factor_names_full(self):
        fm = self.model_pf.factor_model_
        assert len(fm.factor_names) == 4, (
            "All 4 original factor names must be preserved"
        )

    def test_basis_present(self):
        fm = self.model_pf.factor_model_
        assert fm.family_constraint_basis is not None

    def test_factor_names_match(self):
        assert list(self.model_pf.factor_model_.factor_names) == list(
            self.model_full.factor_model_.factor_names
        )

    def test_factor_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_covariance,
            self.model_full.factor_model_.factor_covariance,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.loading_matrix,
            self.model_full.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )


class TestPartialFitIdioCorrelation:
    """partial_fit with idio_corr_threshold > 0 must match fit(all)."""

    N_OBS = 500
    N_ASSETS = 50
    SEED = 777
    PAIR_RHO = 0.5
    THRESHOLD = 0.1

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)
        betas = rng.uniform(0.5, 1.5, size=cls.N_ASSETS)
        sigma_f, sigma_eps = 0.01, 0.005
        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_ASSETS))

        rho = cls.PAIR_RHO
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
        z = rng.normal(0, sigma_eps, size=(cls.N_OBS, 2))
        eps[:, :2] = z @ chol.T

        returns = betas[None, :] * f_true[:, None] + eps
        betas_2d = np.broadcast_to(betas, (cls.N_OBS, cls.N_ASSETS)).copy()
        panel, X = make_panel(returns, extra_fields={"beta": betas_2d})

        model_full = _make_model(idio_corr_threshold=cls.THRESHOLD)
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_model(idio_corr_threshold=cls.THRESHOLD)
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_idio_covariance_not_diagonal(self):
        D = self.model_pf.factor_model_.idio_covariance
        off_diag = D.copy()
        np.fill_diagonal(off_diag, 0.0)
        assert np.any(np.abs(off_diag) > 1e-10)

    def test_idio_covariance_match(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_covariance,
            self.model_full.factor_model_.idio_covariance,
            rtol=1e-10,
        )

    def test_asset_covariance_match(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )


class TestFitAfterPartialFitResets:
    """Calling fit after partial_fit must fully reset state."""

    N_OBS = 200
    N_ASSETS = 50
    SEED = 33

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model = _make_model()
        split = cls.N_OBS // 2
        model.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model.partial_fit(X.iloc[split:], characteristics=panel[split:])

        # Now call fit on just the second half -- must discard first-half state
        model.fit(X.iloc[split:], characteristics=panel[split:])

        model_ref = _make_model()
        model_ref.fit(X.iloc[split:], characteristics=panel[split:])

        cls.model = model
        cls.model_ref = model_ref
        cls.half_len = cls.N_OBS - split

    def test_observations_length(self):
        """After fit, accumulated history must be from the fit call only."""
        assert len(self.model.factor_model_.observations) == len(
            self.model_ref.factor_model_.observations
        )

    def test_factor_returns_shape(self):
        assert (
            self.model.factor_model_.factor_returns.shape
            == self.model_ref.factor_model_.factor_returns.shape
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model.return_distribution_.covariance,
            self.model_ref.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model.factor_model_.loading_matrix,
            self.model_ref.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_factor_returns_values(self):
        np.testing.assert_allclose(
            self.model.factor_model_.factor_returns,
            self.model_ref.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_accumulators_reset(self):
        """After fit, accumulated data must come from the fit call only."""
        assert len(self.model._get_history()["observations"]) == len(
            self.model_ref._get_history()["observations"]
        )


class TestPartialFitSingleObservation:
    """Streaming two-batch partial_fit must match fit(all).

    Uses N_OBS=200 and a midpoint split because the factor prior estimator
    needs enough observations to produce finite mu/covariance; smaller
    chunks yield all-NaN from moment estimators.
    """

    N_OBS = 200
    N_ASSETS = 50
    SEED = 11

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X, *_ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_full = _make_model()
        model_full.fit(X, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_model()
        model_pf.partial_fit(X.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_observations_length(self):
        assert len(self.model_pf.factor_model_.observations) == len(
            self.model_full.factor_model_.observations
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_single_obs_idio_returns(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_returns,
            self.model_full.factor_model_.idio_returns,
            rtol=1e-10,
        )


class TestPartialFitInvestmentUniverse:
    """partial_fit with investment universe (X subset) != coverage must match fit(all).

    The regression runs on the full coverage universe (50 assets) but outputs
    are subsetted to a smaller investment universe (30 assets) via X.
    """

    N_OBS = 200
    N_COVERAGE = 50
    N_INVEST = 30
    SEED = 44

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        rng = np.random.default_rng(cls.SEED)
        betas = rng.uniform(0.5, 1.5, size=cls.N_COVERAGE)
        sigma_f, sigma_eps = 0.01, 0.005

        f_true = rng.normal(0, sigma_f, size=cls.N_OBS)
        eps = rng.normal(0, sigma_eps, size=(cls.N_OBS, cls.N_COVERAGE))
        returns = betas[None, :] * f_true[:, None] + eps

        betas_2d = np.broadcast_to(betas, (cls.N_OBS, cls.N_COVERAGE)).copy()
        all_names = np.array([f"asset_{i}" for i in range(cls.N_COVERAGE)])

        panel, _ = make_panel(
            returns, extra_fields={"beta": betas_2d}, asset_names=all_names
        )
        X_invest = pd.DataFrame(
            returns[:, : cls.N_INVEST], columns=all_names[: cls.N_INVEST]
        )

        model_full = _make_model()
        model_full.fit(X_invest, characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_model()
        model_pf.partial_fit(X_invest.iloc[:split], characteristics=panel[:split])
        model_pf.partial_fit(X_invest.iloc[split:], characteristics=panel[split:])

        cls.model_full = model_full
        cls.model_pf = model_pf

    def test_loading_matrix_shape(self):
        B = self.model_pf.factor_model_.loading_matrix
        assert B.shape[0] == self.N_INVEST

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_pf.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.loading_matrix,
            self.model_full.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_factor_returns_accumulated(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_idio_returns_shape(self):
        assert self.model_pf.factor_model_.idio_returns.shape[1] == self.N_INVEST

    def test_invest_universe_idio_returns(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.idio_returns,
            self.model_full.factor_model_.idio_returns,
            rtol=1e-10,
        )
