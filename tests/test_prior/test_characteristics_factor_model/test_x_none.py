"""Tests for CharacteristicsFactorModel with X=None (full coverage universe).

Verifies that omitting X produces identical results to passing a DataFrame
whose columns match the full coverage universe, and that no unnecessary
subsetting occurs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.prior import CharacteristicsFactorModel

from .conftest import make_panel, passthrough_factor


def _make_single_factor_data(n_obs, n_assets, seed=42):
    rng = np.random.default_rng(seed)
    beta_true = rng.uniform(0.5, 1.5, size=n_assets)
    sigma_f = 0.01
    sigma_eps = rng.uniform(0.005, 0.02, size=n_assets)
    f_true = rng.normal(0, sigma_f, size=n_obs)
    eps = rng.normal(0, 1, size=(n_obs, n_assets)) * sigma_eps
    returns = beta_true[None, :] * f_true[:, None] + eps
    betas_field = np.broadcast_to(beta_true, (n_obs, n_assets)).copy()
    panel, X = make_panel(returns, extra_fields={"beta": betas_field})
    return panel, X


def _make_model(**overrides):
    defaults = dict(
        factors=[("beta", passthrough_factor("beta", family="market"))],
        exposure_lag=1,
        benchmark_mcap_power=0,
        regression_mcap_power=0,
    )
    defaults.update(overrides)
    return CharacteristicsFactorModel(**defaults)


class TestXNoneEquivalence:
    """fit(X=None) must produce identical results to fit(X=full_coverage_df)."""

    N_OBS = 200
    N_ASSETS = 50
    SEED = 42

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_x = _make_model()
        model_x.fit(X, characteristics=panel)

        model_none = _make_model()
        model_none.fit(characteristics=panel)

        cls.model_x = model_x
        cls.model_none = model_none

    def test_asset_mu(self):
        np.testing.assert_allclose(
            self.model_none.return_distribution_.mu,
            self.model_x.return_distribution_.mu,
            rtol=1e-10,
        )

    def test_asset_covariance(self):
        np.testing.assert_allclose(
            self.model_none.return_distribution_.covariance,
            self.model_x.return_distribution_.covariance,
            rtol=1e-10,
        )

    def test_covariance_sqrt(self):
        sqrt_none = self.model_none.return_distribution_.covariance_sqrt
        sqrt_x = self.model_x.return_distribution_.covariance_sqrt
        for b_none, b_x in zip(sqrt_none.components, sqrt_x.components, strict=False):
            np.testing.assert_allclose(b_none, b_x, rtol=1e-10)
        if sqrt_none.diagonal is not None:
            np.testing.assert_allclose(sqrt_none.diagonal, sqrt_x.diagonal, rtol=1e-10)

    def test_systematic_returns(self):
        np.testing.assert_allclose(
            self.model_none.return_distribution_.returns,
            self.model_x.return_distribution_.returns,
            rtol=1e-10,
        )

    def test_loading_matrix(self):
        np.testing.assert_allclose(
            self.model_none.factor_model_.loading_matrix,
            self.model_x.factor_model_.loading_matrix,
            rtol=1e-10,
        )

    def test_factor_covariance(self):
        np.testing.assert_allclose(
            self.model_none.factor_model_.factor_covariance,
            self.model_x.factor_model_.factor_covariance,
            rtol=1e-10,
        )

    def test_idio_covariance(self):
        np.testing.assert_allclose(
            self.model_none.factor_model_.idio_covariance,
            self.model_x.factor_model_.idio_covariance,
            rtol=1e-10,
        )

    def test_asset_names(self):
        np.testing.assert_array_equal(
            self.model_none.factor_model_.asset_names,
            self.model_x.factor_model_.asset_names,
        )

    def test_feature_names_in(self):
        np.testing.assert_array_equal(
            self.model_none.feature_names_in_,
            self.model_x.feature_names_in_,
        )

    def test_n_features_in(self):
        assert self.model_none.n_features_in_ == self.model_x.n_features_in_


class TestXNoneInvestmentIdxInCoverage:
    """When X=None, `_investment_idx_in_coverage` is unset (full universe)."""

    def test_investment_idx_in_coverage_none_when_x_none(self):
        panel, _ = _make_single_factor_data(100, 50)
        model = _make_model()
        model.fit(characteristics=panel)
        assert model._investment_idx_in_coverage is None

    def test_investment_idx_in_coverage_set_when_x_provided(self):
        panel, X = _make_single_factor_data(100, 50)
        model = _make_model()
        model.fit(X, characteristics=panel)
        assert model._investment_idx_in_coverage is not None


class TestXNonePartialFit:
    """partial_fit with X=None must match fit(X=None)."""

    N_OBS = 200
    N_ASSETS = 50
    SEED = 88

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, _ = _make_single_factor_data(cls.N_OBS, cls.N_ASSETS, cls.SEED)

        model_full = _make_model()
        model_full.fit(characteristics=panel)

        split = cls.N_OBS // 2
        model_pf = _make_model()
        model_pf.partial_fit(characteristics=panel[:split])
        model_pf.partial_fit(characteristics=panel[split:])

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

    def test_factor_returns(self):
        np.testing.assert_allclose(
            self.model_pf.factor_model_.factor_returns,
            self.model_full.factor_model_.factor_returns,
            rtol=1e-10,
        )

    def test_observations(self):
        np.testing.assert_array_equal(
            self.model_pf.factor_model_.observations,
            self.model_full.factor_model_.observations,
        )


class TestXNoneSubsetEquivalence:
    """fit(X=subset_df) must produce correctly subsetted outputs."""

    N_OBS = 200
    N_COVERAGE = 50
    N_INVEST = 30
    SEED = 77

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup(cls):
        panel, X = _make_single_factor_data(cls.N_OBS, cls.N_COVERAGE, cls.SEED)
        X_subset = X.iloc[:, : cls.N_INVEST]

        model_full = _make_model()
        model_full.fit(characteristics=panel)

        model_sub = _make_model()
        model_sub.fit(X_subset, characteristics=panel)

        cls.model_full = model_full
        cls.model_sub = model_sub

    def test_subset_shapes(self):
        assert self.model_sub.return_distribution_.mu.shape == (self.N_INVEST,)
        assert self.model_sub.return_distribution_.covariance.shape == (
            self.N_INVEST,
            self.N_INVEST,
        )

    def test_full_shapes(self):
        assert self.model_full.return_distribution_.mu.shape == (self.N_COVERAGE,)

    def test_subset_mu_matches_full_slice(self):
        np.testing.assert_allclose(
            self.model_sub.return_distribution_.mu,
            self.model_full.return_distribution_.mu[: self.N_INVEST],
            rtol=1e-10,
        )

    def test_subset_covariance_matches_full_slice(self):
        np.testing.assert_allclose(
            self.model_sub.return_distribution_.covariance,
            self.model_full.return_distribution_.covariance[
                : self.N_INVEST, : self.N_INVEST
            ],
            rtol=1e-10,
        )


class TestXNoneValidation:
    """Validation edge cases for X=None and X=DataFrame."""

    def test_x_not_dataframe_raises(self):
        panel, _ = _make_single_factor_data(100, 50)
        model = _make_model()
        with pytest.raises(ValueError, match=r"`X` must be a pd\.DataFrame or None"):
            model.fit(np.zeros((100, 50)), characteristics=panel)

    def test_x_none_coverage_asset_names(self):
        panel, _ = _make_single_factor_data(100, 50)
        model = _make_model()
        model.fit(characteristics=panel)
        np.testing.assert_array_equal(model.feature_names_in_, model.asset_names_)

    def test_x_subset_unknown_asset_raises(self):
        panel, _ = _make_single_factor_data(100, 50)
        X_bad = pd.DataFrame(
            np.zeros((100, 3)), columns=["unknown_0", "unknown_1", "unknown_2"]
        )
        model = _make_model()
        with pytest.raises(ValueError, match="missing from"):
            model.fit(X_bad, characteristics=panel)
