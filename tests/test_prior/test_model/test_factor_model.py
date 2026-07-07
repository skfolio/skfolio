"""Tests for FactorModel"""

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from skfolio.containers import AssetPanel
from skfolio.prior import (
    CorrelationMethod,
    FactorModel,
    ReturnDistribution,
)
from skfolio.prior._model._family_constraint_basis import (
    FamilyConstraint,
    FamilyConstraintBasis,
    compute_family_constraint_basis,
)
from skfolio.utils.stats import CSWeighting, cov_to_corr, cs_pearson_correlation


def _make_factor_model(
    n_obs=50,
    n_assets=30,
    n_factors=4,
    seed=42,
    with_time_series=True,
    factor_families=None,
):
    """Build a synthetic FactorModel for diagnostics testing."""
    rng = np.random.default_rng(seed)

    factor_names = np.array([f"factor_{k}" for k in range(n_factors)])
    asset_names = np.array([f"asset_{i}" for i in range(n_assets)])
    observations = pd.date_range("2020-01-01", periods=n_obs, freq="B")

    A = rng.standard_normal((n_factors, n_factors))
    factor_cov = A @ A.T / n_factors

    loading = rng.standard_normal((n_assets, n_factors)) * 0.5
    factor_mu = rng.standard_normal(n_factors) * 0.001
    idio_cov = rng.uniform(0.001, 0.01, size=n_assets)

    if with_time_series:
        factor_returns = rng.multivariate_normal(factor_mu, factor_cov, size=n_obs)
        exposures = np.tile(loading, (n_obs, 1, 1))
        exposures += rng.standard_normal(exposures.shape) * 0.05
        idio_returns = rng.standard_normal((n_obs, n_assets)) * np.sqrt(idio_cov)
        idio_variances = np.tile(idio_cov, (n_obs, 1))
        idio_variances += rng.uniform(0, 0.001, size=(n_obs, n_assets))
    else:
        factor_returns = None
        exposures = None
        idio_returns = None
        idio_variances = None

    if factor_families is not None:
        factor_families = np.asarray(factor_families)

    return FactorModel(
        observations=np.asarray(observations),
        asset_names=asset_names,
        factor_names=factor_names,
        factor_families=factor_families,
        loading_matrix=loading,
        exposures=exposures,
        factor_covariance=factor_cov,
        factor_mu=factor_mu,
        factor_returns=factor_returns,
        idio_covariance=idio_cov,
        idio_mu=None,
        idio_returns=idio_returns,
        idio_variances=idio_variances,
    )


def _make_exposure_only_factor_model(
    exposures,
    benchmark_weights=None,
    regression_weights=None,
):
    """Build a minimal FactorModel for exposure-only diagnostics tests."""
    exposures = np.asarray(exposures, dtype=float)
    n_obs, n_assets, n_factors = exposures.shape
    return FactorModel(
        observations=pd.date_range("2020-01-01", periods=n_obs, freq="B").to_numpy(),
        asset_names=np.array([f"asset_{i}" for i in range(n_assets)]),
        factor_names=np.array([f"factor_{k}" for k in range(n_factors)]),
        factor_families=None,
        loading_matrix=np.nan_to_num(exposures[-1], nan=0.0),
        exposures=exposures,
        factor_covariance=np.eye(n_factors),
        factor_mu=np.zeros(n_factors),
        factor_returns=None,
        idio_covariance=np.ones(n_assets),
        idio_mu=None,
        idio_returns=None,
        idio_variances=None,
        benchmark_weights=benchmark_weights,
        regression_weights=regression_weights,
    )


@pytest.fixture()
def factor_model():
    return _make_factor_model()


@pytest.fixture()
def factor_model_with_families():
    return _make_factor_model(
        n_factors=5,
        factor_families=["style", "style", "style", "industry", "country"],
    )


@pytest.fixture()
def factor_model_with_weights():
    rng = np.random.default_rng(99)
    fm = _make_factor_model()
    n_obs = fm.observations.shape[0]
    n_assets = fm.asset_names.shape[0]
    reg_w = rng.uniform(0.1, 1.0, size=(n_obs, n_assets))
    bmk_w = rng.uniform(0.5, 2.0, size=(n_obs, n_assets))
    bmk_w = bmk_w / bmk_w.sum(axis=1, keepdims=True)
    return FactorModel(
        observations=fm.observations,
        asset_names=fm.asset_names,
        factor_names=fm.factor_names,
        factor_families=fm.factor_families,
        loading_matrix=fm.loading_matrix,
        exposures=fm.exposures,
        factor_covariance=fm.factor_covariance,
        factor_mu=fm.factor_mu,
        factor_returns=fm.factor_returns,
        idio_covariance=fm.idio_covariance,
        idio_mu=fm.idio_mu,
        idio_returns=fm.idio_returns,
        idio_variances=fm.idio_variances,
        regression_weights=reg_w,
        benchmark_weights=bmk_w,
    )


@pytest.fixture()
def factor_model_no_ts():
    return _make_factor_model(with_time_series=False)


@pytest.fixture()
def factor_model_with_basis():
    """FactorModel with a basket-neutral basis (industry constraint)."""
    rng = np.random.default_rng(42)
    T, N = 50, 30
    n_industries = 3
    K = 1 + n_industries + 1  # mkt + industries + style
    factor_names = np.array(
        ["mkt"] + [f"ind{i}" for i in range(n_industries)] + ["style"]
    )
    factor_families = np.array(["market"] + ["industry"] * n_industries + ["style"])
    exposures = rng.standard_normal((T, N, K))
    exposures[:, :, 0] = 1.0
    for i in range(n_industries):
        group = slice(i * (N // n_industries), (i + 1) * (N // n_industries))
        exposures[:, group, 1 + i] = 1.0
        for j in range(n_industries):
            if j != i:
                exposures[:, group, 1 + j] = 0.0
    bench_w = np.ones((T, N)) / N

    basis, _ = compute_family_constraint_basis(
        constrained_families=[("industry", None)],
        factor_exposures=exposures,
        benchmark_weights=bench_w,
        factor_names=factor_names,
        factor_families=factor_families,
    )
    fr_red = rng.standard_normal((T, basis.n_reduced_factors)) * 0.01
    fr_full = basis.expand_factor_returns(fr_red)
    ir = rng.standard_normal((T, N)) * 0.05
    A = rng.standard_normal((K, K))

    return FactorModel(
        observations=pd.date_range("2020-01-01", periods=T, freq="B").to_numpy(),
        asset_names=np.array([f"asset_{i}" for i in range(N)]),
        factor_names=factor_names,
        factor_families=factor_families,
        loading_matrix=exposures[-1],
        exposures=exposures,
        factor_covariance=A @ A.T / K,
        factor_mu=np.zeros(K),
        factor_returns=fr_full,
        idio_covariance=np.ones(N) * 0.01,
        idio_mu=None,
        idio_returns=ir,
        idio_variances=None,
        exposure_lag=1,
        benchmark_weights=bench_w,
        family_constraint_basis=basis,
    )


# ------------------------------------------------------------------
# Summary tables
# ------------------------------------------------------------------
_SUMMARY_COLUMNS = {
    "annualized_mean",
    "annualized_vol",
    "annualized_sharpe",
    "autocorrelation",
    "mean_abs_t_stat",
    "t_stat_exceedance_rate",
    "mean_vif",
    "stability",
    "coverage",
}


class TestDataFrameAccessors:
    def test_factor_returns_df_requires_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="factor_returns"):
            _ = factor_model_no_ts.factor_returns_df

    def test_idio_returns_df_requires_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_returns_df

    def test_exposures_df_requires_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            _ = factor_model_no_ts.exposures_df


class TestEnrichAssetPanel:
    def test_aligns_by_labels_and_shallow_copies(self, factor_model_with_weights):
        fm = factor_model_with_weights
        observations = np.concatenate([[np.datetime64("2019-12-31")], fm.observations])
        asset_names = fm.asset_names[::-1]
        panel = AssetPanel(
            fields={"returns": np.zeros((len(observations), len(asset_names)))},
            observations=observations,
            asset_names=asset_names,
        )

        enriched = fm.enrich_asset_panel(panel)

        assert enriched is not panel
        assert "idio_returns" not in panel.fields
        assert {
            "idio_returns",
            "idio_variances",
            "regression_weights",
            "exposures",
        }.issubset(enriched.fields)
        assert np.isnan(enriched["idio_returns"][0]).all()
        assert np.isnan(enriched["idio_variances"][0]).all()
        assert (enriched["regression_weights"][0] == 0.0).all()
        assert np.isnan(enriched["exposures"][0]).all()
        np.testing.assert_allclose(
            enriched["idio_returns"][1:], fm.idio_returns[:, ::-1]
        )
        np.testing.assert_allclose(
            enriched["idio_variances"][1:], fm.idio_variances[:, ::-1]
        )
        np.testing.assert_allclose(
            enriched["regression_weights"][1:], fm.regression_weights[:, ::-1]
        )
        np.testing.assert_allclose(enriched["exposures"][1:], fm.exposures[:, ::-1])
        np.testing.assert_array_equal(
            enriched.fields["exposures"].third_axis_labels, fm.factor_names
        )

    def test_copy_false_mutates_panel(self, factor_model_with_weights):
        fm = factor_model_with_weights
        panel = AssetPanel(
            fields={"returns": np.zeros((len(fm.observations), len(fm.asset_names)))},
            observations=fm.observations,
            asset_names=fm.asset_names,
        )

        enriched = fm.enrich_asset_panel(panel, copy=False)

        assert enriched is panel
        np.testing.assert_allclose(panel["idio_returns"], fm.idio_returns)

    def test_enriches_view_with_view_local_fields(self, factor_model_with_weights):
        fm = factor_model_with_weights
        panel = AssetPanel(
            fields={"returns": np.zeros((len(fm.observations), len(fm.asset_names)))},
            observations=fm.observations,
            asset_names=fm.asset_names,
        )
        view = panel[5:]

        enriched = fm.enrich_asset_panel(view)

        assert enriched is not view
        assert enriched.owner is view.owner
        assert "idio_returns" not in view.fields
        assert "idio_returns" not in panel.fields
        assert "idio_returns" in enriched.fields
        np.testing.assert_allclose(enriched["idio_returns"], fm.idio_returns[5:])
        np.testing.assert_allclose(enriched["exposures"], fm.exposures[5:])

    def test_copy_false_mutates_view_locally(self, factor_model_with_weights):
        fm = factor_model_with_weights
        panel = AssetPanel(
            fields={"returns": np.zeros((len(fm.observations), len(fm.asset_names)))},
            observations=fm.observations,
            asset_names=fm.asset_names,
        )
        view = panel[5:]

        enriched = fm.enrich_asset_panel(view, copy=False)

        assert enriched is view
        assert "idio_returns" in view.fields
        assert "idio_returns" not in panel.fields
        np.testing.assert_allclose(view["idio_returns"], fm.idio_returns[5:])

    def test_uses_effective_exposures(self, factor_model_with_basis):
        fm = replace(
            factor_model_with_basis,
            idio_variances=np.ones_like(factor_model_with_basis.idio_returns),
            regression_weights=np.ones_like(factor_model_with_basis.idio_returns),
        )
        panel = AssetPanel(
            fields={"returns": np.zeros((len(fm.observations), len(fm.asset_names)))},
            observations=fm.observations,
            asset_names=fm.asset_names,
        )

        enriched = fm.enrich_asset_panel(panel)

        expected = fm.family_constraint_basis.reduce_exposures(fm.exposures)
        np.testing.assert_allclose(enriched["exposures"], expected)
        np.testing.assert_array_equal(
            enriched.fields["exposures"].third_axis_labels,
            fm.effective_factor_names,
        )
        np.testing.assert_array_equal(
            enriched.fields["exposures"].third_axis_groups,
            fm.effective_factor_families,
        )

    def test_raises_when_observations_do_not_overlap(self, factor_model_with_weights):
        factor_model = factor_model_with_weights
        observations = np.array(
            [np.datetime64("2019-01-01"), np.datetime64("2019-01-02")]
        )
        panel = AssetPanel(
            fields={
                "returns": np.zeros((len(observations), len(factor_model.asset_names)))
            },
            observations=observations,
            asset_names=factor_model.asset_names,
        )

        with pytest.raises(ValueError, match="observations do not overlap"):
            factor_model.enrich_asset_panel(panel)

    def test_raises_when_assets_do_not_match(self, factor_model_with_weights):
        factor_model = factor_model_with_weights
        panel = AssetPanel(
            fields={
                "returns": np.zeros(
                    (len(factor_model.observations), len(factor_model.asset_names) - 1)
                )
            },
            observations=factor_model.observations,
            asset_names=factor_model.asset_names[:-1],
        )

        with pytest.raises(ValueError, match="asset names"):
            factor_model.enrich_asset_panel(panel)

    def test_raises_when_field_exists(self, factor_model):
        shape = (len(factor_model.observations), len(factor_model.asset_names))
        panel = AssetPanel(
            fields={
                "returns": np.zeros(shape),
                "idio_returns": np.zeros(shape),
            },
            observations=factor_model.observations,
            asset_names=factor_model.asset_names,
        )

        with pytest.raises(ValueError, match="already contains"):
            factor_model.enrich_asset_panel(panel)


class TestSummary:
    def test_shape_and_columns(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 4
        assert set(df.columns) == _SUMMARY_COLUMNS

    def test_requires_factor_returns(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="factor_returns"):
            factor_model_no_ts.summary()

    def test_annualized_stats(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert df["annualized_vol"].gt(0).all()
        assert df["annualized_sharpe"].notna().all()

    def test_autocorrelation_in_range(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert (df["autocorrelation"] >= -1.0).all()
        assert (df["autocorrelation"] <= 1.0).all()

    def test_mean_abs_t_stat_positive(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        valid = df["mean_abs_t_stat"].dropna()
        assert (valid >= 0).all()

    def test_t_stat_exceedance_rate_in_unit_interval(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        valid = df["t_stat_exceedance_rate"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_custom_t_stat_threshold(self, factor_model):
        df_low = factor_model.summary(
            t_stat_threshold=1.0, stability_cs_weighting=CSWeighting.IDENTITY
        )
        df_high = factor_model.summary(
            t_stat_threshold=3.0, stability_cs_weighting=CSWeighting.IDENTITY
        )
        valid_low = df_low["t_stat_exceedance_rate"].dropna()
        valid_high = df_high["t_stat_exceedance_rate"].dropna()
        assert (valid_low >= valid_high).all()

    def test_mean_vif_at_least_one(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        valid = df["mean_vif"].dropna()
        assert (valid >= 1.0 - 1e-6).all()

    def test_stability_in_correlation_range(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        valid = df["stability"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_coverage_in_unit_interval(self, factor_model):
        df = factor_model.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert (df["coverage"] >= 0).all()
        assert (df["coverage"] <= 1).all()

    def test_coverage_respects_regression_weights(self):
        """Coverage should count only estimation-universe assets."""
        rng = np.random.default_rng(7)
        n_obs, n_assets, n_factors = 50, 30, 2
        factor_names = np.array([f"f{k}" for k in range(n_factors)])
        asset_names = np.array([f"a{i}" for i in range(n_assets)])
        obs = pd.date_range("2020-01-01", periods=n_obs, freq="B").to_numpy()

        exposures = rng.standard_normal((n_obs, n_assets, n_factors))
        loading = exposures[-1]
        A = rng.standard_normal((n_factors, n_factors))
        fr = rng.standard_normal((n_obs, n_factors)) * 0.01
        ir = rng.standard_normal((n_obs, n_assets)) * 0.05

        reg_w = np.ones((n_obs, n_assets))
        reg_w[:, n_assets // 2 :] = 0.0

        fm = FactorModel(
            observations=obs,
            asset_names=asset_names,
            factor_names=factor_names,
            factor_families=None,
            loading_matrix=loading,
            exposures=exposures,
            factor_covariance=A @ A.T / n_factors,
            factor_mu=np.zeros(n_factors),
            factor_returns=fr,
            idio_covariance=np.ones(n_assets) * 0.01,
            idio_mu=None,
            idio_returns=ir,
            idio_variances=None,
            regression_weights=reg_w,
        )
        df = fm.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert (df["coverage"] == 1.0).all()

        fm_no_w = FactorModel(
            observations=obs,
            asset_names=asset_names,
            factor_names=factor_names,
            factor_families=None,
            loading_matrix=loading,
            exposures=exposures,
            factor_covariance=A @ A.T / n_factors,
            factor_mu=np.zeros(n_factors),
            factor_returns=fr,
            idio_covariance=np.ones(n_assets) * 0.01,
            idio_mu=None,
            idio_returns=ir,
            idio_variances=None,
        )
        assert (
            fm_no_w.summary(stability_cs_weighting=CSWeighting.IDENTITY)["coverage"]
            == 1.0
        ).all()

    def test_custom_stability_step(self, factor_model):
        df = factor_model.summary(
            stability_step=5, stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert df["stability"].notna().any()

    def test_step_too_large_returns_nan_stability(self, factor_model):
        df = factor_model.summary(
            stability_step=100, stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert df["stability"].isna().all()

    def test_families_filter(self, factor_model_with_families):
        df = factor_model_with_families.summary(
            families="style", stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert df.shape[0] == 3
        assert list(df.index) == ["factor_0", "factor_1", "factor_2"]

    def test_passes_factor_subset_to_exposure_stability(
        self, factor_model, monkeypatch
    ):
        captured = {}

        def fake_exposure_stability(
            self,
            exposures,
            step=21,
            cs_weighting=CSWeighting.BENCHMARK,
        ):
            captured["n_factors"] = exposures.shape[2]
            return np.zeros((exposures.shape[0] - step, exposures.shape[2]))

        monkeypatch.setattr(FactorModel, "_exposure_stability", fake_exposure_stability)

        factor_model.summary(
            factors=["factor_0", "factor_2"],
            stability_cs_weighting=CSWeighting.IDENTITY,
        )

        assert captured["n_factors"] == 2

    def test_families_none_returns_all(self, factor_model_with_families):
        df = factor_model_with_families.summary(
            families=None, stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert df.shape[0] == 5

    def test_families_list(self, factor_model_with_families):
        df = factor_model_with_families.summary(
            families=["style", "country"],
            stability_cs_weighting=CSWeighting.IDENTITY,
        )
        assert df.shape[0] == 4

    def test_families_unknown_raises(self, factor_model_with_families):
        with pytest.raises(ValueError, match="Unknown"):
            factor_model_with_families.summary(families="unknown")

    def test_families_raises_when_none_in_model(self, factor_model):
        with pytest.raises(ValueError):
            factor_model.summary(families="style")

    def test_stability_weighting(self, factor_model_with_weights):
        df_bmk = factor_model_with_weights.summary(
            stability_cs_weighting=CSWeighting.BENCHMARK
        )
        df_eq = factor_model_with_weights.summary(
            stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert not np.allclose(
            df_bmk["stability"].values,
            df_eq["stability"].values,
            atol=1e-6,
        )

    def test_stability_default_warns_when_benchmark_missing(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            factor_model.summary()

    def test_gram_diagnostics_with_basis(self, factor_model_with_basis):
        df = factor_model_with_basis.summary(
            families=None, stability_cs_weighting=CSWeighting.IDENTITY
        )
        assert df.shape[0] == 5
        gram_names = set(factor_model_with_basis._reduced_factor_names)
        for name in df.index:
            if name in gram_names:
                assert np.isfinite(df.loc[name, "mean_abs_t_stat"])
            else:
                assert np.isnan(df.loc[name, "mean_abs_t_stat"])

    def test_gram_errors_propagate(self, factor_model, monkeypatch):
        def _raise(_self):
            raise RuntimeError("boom")

        monkeypatch.setattr(FactorModel, "_gram_diagnostics", property(_raise))
        with pytest.raises(RuntimeError, match="boom"):
            factor_model.summary()


class TestExposureICSummary:
    def test_shape_and_columns(self, factor_model):
        df = factor_model.exposure_ic_summary()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)
        assert set(df.columns) == {
            "mean_ic",
            "std_ic",
            "ic_ir",
            "hit_rate",
        }

    def test_spearman_ic_default(self, factor_model):
        df_spearman = factor_model.exposure_ic_summary(
            correlation_method=CorrelationMethod.SPEARMAN
        )
        df_pearson = factor_model.exposure_ic_summary(
            correlation_method=CorrelationMethod.PEARSON
        )
        assert not np.allclose(
            df_spearman["mean_ic"].values,
            df_pearson["mean_ic"].values,
        )

    def test_ic_in_range(self, factor_model):
        df = factor_model.exposure_ic_summary()
        assert (df["mean_ic"] >= -1.0).all()
        assert (df["mean_ic"] <= 1.0).all()

    def test_hit_rate_in_unit_interval(self, factor_model):
        df = factor_model.exposure_ic_summary()
        assert (df["hit_rate"] >= 0.0).all()
        assert (df["hit_rate"] <= 1.0).all()

    def test_families_filter(self, factor_model_with_families):
        df = factor_model_with_families.exposure_ic_summary(families="style")
        assert df.shape[0] == 3

    def test_passes_factor_subset_to_ic(self, factor_model, monkeypatch):
        captured = {}

        def fake_ic(
            self,
            correlation_method=CorrelationMethod.SPEARMAN,
            horizon=1,
            factor_indices=None,
            reduced_basis=False,
        ):
            captured["factor_indices"] = factor_indices
            n_selected = len(factor_indices) if factor_indices is not None else 4
            return np.zeros((3, n_selected)), 0

        monkeypatch.setattr(FactorModel, "_ic", fake_ic)

        factor_model.exposure_ic_summary(factors=["factor_0", "factor_2"])

        assert captured["factor_indices"] == [0, 2]

    def test_requires_time_series(self, factor_model_no_ts):
        with pytest.raises(ValueError):
            factor_model_no_ts.exposure_ic_summary()

    def test_custom_horizon(self, factor_model):
        df = factor_model.exposure_ic_summary(horizon=5)
        assert df.shape == (4, 4)
        assert (df["mean_ic"] >= -1.0).all()
        assert (df["mean_ic"] <= 1.0).all()

    def test_horizon_too_large_raises(self, factor_model):
        with pytest.raises(ValueError, match="Not enough observations"):
            factor_model.exposure_ic_summary(horizon=100)

    def test_horizon_zero_raises(self, factor_model):
        with pytest.raises(ValueError, match="must be >= 1"):
            factor_model.exposure_ic_summary(horizon=0)

    def test_spearman_ic_uses_joint_finite_ranks(self):
        exposures = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[1.0], [2.0], [3.0], [4.0]],
                [[1.0], [2.0], [3.0], [4.0]],
            ]
        )
        idio_returns = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [40.0, np.nan, 20.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        fm = FactorModel(
            observations=np.arange(3),
            asset_names=np.array(["a0", "a1", "a2", "a3"]),
            factor_names=np.array(["factor"]),
            factor_families=None,
            loading_matrix=exposures[-1],
            exposures=exposures,
            factor_covariance=np.eye(1),
            factor_mu=np.zeros(1),
            factor_returns=np.zeros((3, 1)),
            idio_covariance=np.ones(4),
            idio_mu=None,
            idio_returns=idio_returns,
            idio_variances=None,
            exposure_lag=0,
        )

        ic, _ = fm._ic(correlation_method=CorrelationMethod.SPEARMAN, horizon=1)

        assert ic[0, 0] == pytest.approx(-1.0)


class TestPlotCumulativeExposureIC:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_cumulative_exposure_ic()
        assert isinstance(fig, go.Figure)
        assert "Cumulative" in fig.layout.title.text

    def test_pearson(self, factor_model):
        fig = factor_model.plot_cumulative_exposure_ic(
            correlation_method=CorrelationMethod.PEARSON
        )
        assert isinstance(fig, go.Figure)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_cumulative_exposure_ic(factors=["factor_0", "factor_1"])
        assert len(fig.data) == 2

    def test_families_filter(self, factor_model_with_families):
        fig = factor_model_with_families.plot_cumulative_exposure_ic(families="style")
        assert len(fig.data) == 3

    def test_passes_family_subset_to_ic(self, factor_model_with_families, monkeypatch):
        captured = {}

        def fake_ic(
            self,
            correlation_method=CorrelationMethod.SPEARMAN,
            horizon=1,
            factor_indices=None,
            reduced_basis=False,
        ):
            captured["factor_indices"] = factor_indices
            captured["reduced_basis"] = reduced_basis
            n_selected = len(factor_indices) if factor_indices is not None else 4
            return np.zeros((3, n_selected)), 1

        monkeypatch.setattr(FactorModel, "_ic", fake_ic)

        factor_model_with_families.plot_cumulative_exposure_ic(families="style")

        assert captured["factor_indices"] == [0, 1, 2]
        assert captured["reduced_basis"]

    def test_reduced_basis_ic_shape(self, factor_model_with_basis):
        fm = factor_model_with_basis

        ic, _ = fm._ic(
            correlation_method=CorrelationMethod.SPEARMAN,
            horizon=1,
            reduced_basis=True,
        )

        assert ic.shape[1] == fm.family_constraint_basis.n_reduced_factors

    def test_requires_time_series(self, factor_model_no_ts):
        with pytest.raises(ValueError):
            factor_model_no_ts.plot_cumulative_exposure_ic()


class TestSpecificRiskCalibrationSummary:
    def test_returns_series(self, factor_model):
        s = factor_model.idio_calibration_summary()
        assert isinstance(s, pd.Series)
        assert s.name == "idio_calibration"

    def test_index_labels(self, factor_model):
        s = factor_model.idio_calibration_summary()
        assert set(s.index) == {
            "mean_cs_std",
            "median_cs_std",
            "mean_cs_excess_kurtosis",
            "mean_cs_skewness",
            "mean_tail_rate_3sigma",
        }

    def test_std_near_one(self, factor_model):
        s = factor_model.idio_calibration_summary()
        assert 0.5 < s["mean_cs_std"] < 2.0

    def test_tail_rate_non_negative(self, factor_model):
        s = factor_model.idio_calibration_summary()
        assert s["mean_tail_rate_3sigma"] >= 0.0

    def test_requires_idio_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="idio_returns"):
            factor_model_no_ts.idio_calibration_summary()


class TestIdioCalibrationSeries:
    def test_calibration_returns_series(self, factor_model):
        s = factor_model.idio_calibration
        assert isinstance(s, pd.Series)
        assert s.name == "Standardised Idio Return Std"

    def test_calibration_is_cached(self, factor_model):
        assert factor_model.idio_calibration is factor_model.idio_calibration

    def test_kurtosis_returns_series(self, factor_model):
        s = factor_model.idio_kurtosis
        assert isinstance(s, pd.Series)
        assert s.name == "Excess Kurtosis"

    def test_skewness_returns_series(self, factor_model):
        s = factor_model.idio_skewness
        assert isinstance(s, pd.Series)
        assert s.name == "Skewness"

    def test_tail_rate_returns_series(self, factor_model):
        s = factor_model.idio_tail_rate()
        assert isinstance(s, pd.Series)
        assert s.name == "Tail Rate"

    def test_tail_rate_threshold_changes_values(self, factor_model):
        s_high = factor_model.idio_tail_rate(threshold=3.0)
        s_low = factor_model.idio_tail_rate(threshold=2.0)
        assert (s_low >= s_high).all()

    def test_idio_vol_ic_returns_series(self, factor_model):
        s = factor_model.idio_vol_ic
        assert isinstance(s, pd.Series)
        assert s.name == "Idio Vol IC (Spearman)"

    def test_idio_vol_ic_is_cached(self, factor_model):
        assert factor_model.idio_vol_ic is factor_model.idio_vol_ic

    def test_idio_vol_residual_dependence_returns_series(self, factor_model):
        s = factor_model.idio_vol_residual_dependence
        assert isinstance(s, pd.Series)
        assert s.name == "Idio Vol Residual Dependence (Spearman)"

    def test_idio_vol_residual_dependence_is_cached(self, factor_model):
        assert (
            factor_model.idio_vol_residual_dependence
            is factor_model.idio_vol_residual_dependence
        )

    def test_requires_idio_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_calibration
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_kurtosis
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_skewness
        with pytest.raises(ValueError, match="idio_returns"):
            factor_model_no_ts.idio_tail_rate()
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_vol_ic
        with pytest.raises(ValueError, match="idio_returns"):
            _ = factor_model_no_ts.idio_vol_residual_dependence


# ------------------------------------------------------------------
# Plots: factor covariance (no time series needed)
# ------------------------------------------------------------------
class TestPlotFactorForecastCorrelation:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_factor_forecast_correlation()
        assert isinstance(fig, go.Figure)

    def test_works_without_ts(self, factor_model_no_ts):
        fig = factor_model_no_ts.plot_factor_forecast_correlation()
        assert isinstance(fig, go.Figure)

    def test_family_outlines(self, factor_model_with_families):
        fig = factor_model_with_families.plot_factor_forecast_correlation()
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 3

    def test_no_outline_single_family(self, factor_model_with_families):
        fig = factor_model_with_families.plot_factor_forecast_correlation(
            families="style"
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 0

    def test_no_outline_without_families(self, factor_model):
        fig = factor_model.plot_factor_forecast_correlation()
        assert len(fig.layout.shapes) == 0

    def test_no_outline_non_contiguous_families(self):
        fm = _make_factor_model(
            n_factors=4,
            factor_families=["style", "industry", "style", "industry"],
        )
        fig = fm.plot_factor_forecast_correlation()
        assert len(fig.layout.shapes) == 0

    def test_plot_matches_factor_forecast_correlation(self, factor_model):
        fig = factor_model.plot_factor_forecast_correlation()
        z = np.asarray(fig.data[0].z, dtype=float)
        expected = factor_model.factor_forecast_correlation()
        np.testing.assert_allclose(z, expected, atol=1e-12)


class TestFactorForecastCorrelation:
    def test_returns_symmetric_matrix(self, factor_model):
        corr = factor_model.factor_forecast_correlation()
        assert corr.shape == (len(factor_model.factor_names),) * 2
        np.testing.assert_allclose(corr, corr.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)

    def test_works_without_ts(self, factor_model_no_ts):
        corr = factor_model_no_ts.factor_forecast_correlation()
        assert corr.shape == (len(factor_model_no_ts.factor_names),) * 2

    def test_factor_subset(self, factor_model):
        corr = factor_model.factor_forecast_correlation(
            factors=["factor_0", "factor_1"]
        )
        assert corr.shape == (2, 2)

    def test_matches_cov_to_corr(self, factor_model):
        expected, _ = cov_to_corr(factor_model.factor_covariance)
        np.testing.assert_allclose(
            factor_model.factor_forecast_correlation(), expected, atol=1e-12
        )


class TestPlotFactorForecastVolatilities:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_factor_forecast_volatilities()
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, factor_model):
        fig = factor_model.plot_factor_forecast_volatilities(title="Vols")
        assert fig.layout.title.text == "Vols"

    def test_matches_factor_covariance_diag(self, factor_model):
        annualization_factor = 252.0
        fig = factor_model.plot_factor_forecast_volatilities(
            annualization_factor=annualization_factor
        )
        plotted = np.sort(np.asarray(fig.data[0].x, dtype=float))
        expected = np.sort(
            np.sqrt(np.diag(factor_model.factor_covariance) * annualization_factor)
        )
        np.testing.assert_allclose(plotted, expected, atol=1e-12)


# ------------------------------------------------------------------
# Plots: exposure (time series required)
# ------------------------------------------------------------------
class TestPlotExposureDispersion:
    def test_returns_figure(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            fig = factor_model.plot_exposure_dispersion(families=None)
        assert isinstance(fig, go.Figure)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_exposure_dispersion(
            factors=["factor_0", "factor_1"], cs_weighting=CSWeighting.IDENTITY
        )
        assert len(fig.data) == 2

    def test_families_filter(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_dispersion(
            families="style", cs_weighting=CSWeighting.IDENTITY
        )
        assert len(fig.data) == 3

    def test_factors_overrides_families(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_dispersion(
            factors=["factor_3"],
            families="style",
            cs_weighting=CSWeighting.IDENTITY,
        )
        assert len(fig.data) == 1

    def test_unknown_factor(self, factor_model):
        with pytest.raises(ValueError, match="Unknown"):
            factor_model.plot_exposure_dispersion(factors=["bad"])

    def test_requires_exposures(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            factor_model_no_ts.plot_exposure_dispersion()

    def test_uniform_benchmark_weights_match_equal_weights(self):
        exposures = np.array(
            [
                [[1.0, 2.0], [2.0, 0.0], [3.0, -1.0]],
                [[2.0, 1.0], [4.0, 1.0], [6.0, 3.0]],
            ]
        )
        benchmark_weights = np.full(exposures.shape[:2], 1.0 / exposures.shape[1])
        fm = _make_exposure_only_factor_model(
            exposures=exposures,
            benchmark_weights=benchmark_weights,
        )

        fig_bmk = fm.plot_exposure_dispersion(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )
        fig_eq = fm.plot_exposure_dispersion(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )

        for trace_bmk, trace_eq in zip(fig_bmk.data, fig_eq.data, strict=True):
            np.testing.assert_allclose(trace_bmk.y, trace_eq.y, atol=1e-12)


class TestPlotExposureStability:
    def test_returns_figure(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            fig = factor_model.plot_exposure_stability(families=None)
        assert isinstance(fig, go.Figure)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_exposure_stability(
            factors=["factor_0", "factor_1"], cs_weighting=CSWeighting.IDENTITY
        )
        assert len(fig.data) == 2

    def test_custom_step(self, factor_model):
        fig = factor_model.plot_exposure_stability(
            families=None, step=5, cs_weighting=CSWeighting.IDENTITY
        )
        assert isinstance(fig, go.Figure)

    def test_step_too_large_raises(self, factor_model):
        with pytest.raises(ValueError, match="Not enough observations"):
            factor_model.plot_exposure_stability(families=None, step=100)

    def test_with_regression_weights(self, factor_model_with_weights):
        fig = factor_model_with_weights.plot_exposure_stability(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        assert isinstance(fig, go.Figure)

    def test_requires_exposures(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            factor_model_no_ts.plot_exposure_stability()

    def test_families_filter(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_stability(
            families="style", cs_weighting=CSWeighting.IDENTITY
        )
        assert len(fig.data) == 3


class TestPlotExposureCorrelation:
    def test_returns_figure(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            fig = factor_model.plot_exposure_correlation(families=None)
        assert isinstance(fig, go.Figure)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_exposure_correlation(
            factors=["factor_0", "factor_1"], cs_weighting=CSWeighting.IDENTITY
        )
        assert isinstance(fig, go.Figure)

    def test_families_filter(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_correlation(
            families="style", cs_weighting=CSWeighting.IDENTITY
        )
        assert isinstance(fig, go.Figure)

    def test_family_outlines(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_correlation(
            cs_weighting=CSWeighting.IDENTITY
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 3

    def test_no_outline_single_family(self, factor_model_with_families):
        fig = factor_model_with_families.plot_exposure_correlation(
            families="style", cs_weighting=CSWeighting.IDENTITY
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 0

    def test_no_outline_without_families(self, factor_model):
        fig = factor_model.plot_exposure_correlation(cs_weighting=CSWeighting.IDENTITY)
        assert len(fig.layout.shapes) == 0

    def test_no_outline_non_contiguous_families(self):
        fm = _make_factor_model(
            n_factors=4,
            factor_families=["style", "industry", "style", "industry"],
        )
        fig = fm.plot_exposure_correlation(cs_weighting=CSWeighting.IDENTITY)
        assert len(fig.layout.shapes) == 0

    def test_plot_matches_exposure_correlation(self, factor_model):
        fig = factor_model.plot_exposure_correlation(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        z = np.asarray(fig.data[0].z, dtype=float)
        expected = factor_model.exposure_correlation(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        np.testing.assert_allclose(z, expected, atol=1e-12)


class TestExposureCorrelation:
    def test_returns_symmetric_matrix(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            corr = factor_model.exposure_correlation(families=None)
        assert corr.shape == (len(factor_model.factor_names),) * 2
        np.testing.assert_allclose(corr, corr.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)

    def test_factor_subset(self, factor_model):
        corr = factor_model.exposure_correlation(
            factors=["factor_0", "factor_1"], cs_weighting=CSWeighting.IDENTITY
        )
        assert corr.shape == (2, 2)

    def test_requires_exposures(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            factor_model_no_ts.exposure_correlation()

    def test_pairwise_correlation_uses_joint_valid_support(self):
        exposures = np.array(
            [
                [[1.0, 1.0], [2.0, np.nan], [3.0, 5.0], [4.0, 7.0]],
                [[1.0, 3.0], [2.0, 4.0], [np.nan, 5.0], [4.0, 6.0]],
            ]
        )
        benchmark_weights = np.full(exposures.shape[:2], 1.0 / exposures.shape[1])
        fm = _make_exposure_only_factor_model(
            exposures=exposures,
            benchmark_weights=benchmark_weights,
        )

        expected = np.nanmean(
            [
                cs_pearson_correlation(
                    exposures[t, :, 0],
                    exposures[t, :, 1],
                    weights=benchmark_weights[t],
                    axis=0,
                )
                for t in range(exposures.shape[0])
            ]
        )

        corr = fm.exposure_correlation(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )

        np.testing.assert_allclose(corr[0, 1], expected, atol=1e-12)
        np.testing.assert_allclose(corr[1, 0], expected, atol=1e-12)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)


# ------------------------------------------------------------------
# Plots: residual
# ------------------------------------------------------------------
class TestPlotResidualCalibration:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_calibration()
        assert isinstance(fig, go.Figure)

    def test_plot_uses_calibration_series(self, factor_model):
        fig = factor_model.plot_idio_calibration()
        np.testing.assert_allclose(fig.data[0].y, factor_model.idio_calibration.values)

    def test_rolling_window(self, factor_model):
        fig = factor_model.plot_idio_calibration(window=10)
        assert isinstance(fig, go.Figure)

    def test_requires_idio_data(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="idio_returns"):
            factor_model_no_ts.plot_idio_calibration()


class TestPlotResidualTailRate:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_tail_rate()
        assert isinstance(fig, go.Figure)

    def test_plot_uses_tail_rate_series(self, factor_model):
        fig = factor_model.plot_idio_tail_rate()
        np.testing.assert_allclose(fig.data[0].y, factor_model.idio_tail_rate().values)

    def test_custom_threshold(self, factor_model):
        fig = factor_model.plot_idio_tail_rate(threshold=2.0)
        assert "2.0" in fig.layout.title.text


class TestPlotIdioKurtosis:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_kurtosis()
        assert isinstance(fig, go.Figure)

    def test_plot_uses_kurtosis_series(self, factor_model):
        fig = factor_model.plot_idio_kurtosis()
        np.testing.assert_allclose(fig.data[0].y, factor_model.idio_kurtosis.values)

    def test_single_trace(self, factor_model):
        fig = factor_model.plot_idio_kurtosis()
        assert len(fig.data) == 1


class TestPlotIdioSkewness:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_skewness()
        assert isinstance(fig, go.Figure)

    def test_plot_uses_skewness_series(self, factor_model):
        fig = factor_model.plot_idio_skewness()
        np.testing.assert_allclose(fig.data[0].y, factor_model.idio_skewness.values)

    def test_single_trace(self, factor_model):
        fig = factor_model.plot_idio_skewness()
        assert len(fig.data) == 1


class TestPlotIdioVolIc:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_vol_ic(window=10)
        assert isinstance(fig, go.Figure)

    def test_plot_uses_ic_series(self, factor_model):
        fig = factor_model.plot_idio_vol_ic(window=10)
        np.testing.assert_allclose(fig.data[0].y, factor_model.idio_vol_ic.values)

    def test_two_traces(self, factor_model):
        fig = factor_model.plot_idio_vol_ic(window=10)
        assert len(fig.data) == 2


class TestPlotIdioVolResidualDependence:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_idio_vol_residual_dependence(window=10)
        assert isinstance(fig, go.Figure)

    def test_plot_uses_residual_dependence_series(self, factor_model):
        fig = factor_model.plot_idio_vol_residual_dependence(window=10)
        np.testing.assert_allclose(
            fig.data[0].y,
            factor_model.idio_vol_residual_dependence.values,
        )

    def test_two_traces(self, factor_model):
        fig = factor_model.plot_idio_vol_residual_dependence(window=10)
        assert len(fig.data) == 2


class TestWeightingParameter:
    """Verify that `cs_weighting` selects the expected exposure weights."""

    def test_resolve_cs_weighting_benchmark(self, factor_model_with_weights):
        fm = factor_model_with_weights
        w = fm._resolve_cs_weighting(CSWeighting.BENCHMARK, latest=False)
        assert w is fm.benchmark_weights

    def test_resolve_cs_weighting_regression(self, factor_model_with_weights):
        fm = factor_model_with_weights
        w = fm._resolve_cs_weighting(CSWeighting.REGRESSION, latest=False)
        assert w is fm.regression_weights

    def test_resolve_cs_weighting_identity(self, factor_model_with_weights):
        fm = factor_model_with_weights
        assert fm._resolve_cs_weighting(CSWeighting.IDENTITY, latest=False) is None

    def test_resolve_cs_weighting_inverse_idio_variance_panel(self, factor_model):
        weights = factor_model._resolve_cs_weighting(
            CSWeighting.INVERSE_IDIO_VARIANCE, latest=False
        )
        np.testing.assert_allclose(weights, 1.0 / factor_model.idio_variances)

    def test_resolve_cs_weighting_inverse_idio_variance_latest(self, factor_model):
        weights = factor_model._resolve_cs_weighting(
            CSWeighting.INVERSE_IDIO_VARIANCE, latest=True
        )
        np.testing.assert_allclose(weights, 1.0 / factor_model.idio_covariance)

    def test_resolve_cs_weighting_inverse_idio_variance_panel_requires_series(self):
        fm = _make_factor_model(with_time_series=False)
        with pytest.raises(ValueError, match="idio_variances"):
            fm._resolve_cs_weighting(CSWeighting.INVERSE_IDIO_VARIANCE, latest=False)

    @pytest.mark.parametrize("field_name", ["benchmark_weights", "regression_weights"])
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
    def test_invalid_weight_arrays_raise(
        self,
        factor_model_with_weights,
        field_name,
        bad_value,
    ):
        fm = factor_model_with_weights
        weights = getattr(fm, field_name).copy()
        weights[0, 0] = bad_value
        with pytest.raises(ValueError, match=field_name):
            replace(fm, **{field_name: weights})

    def test_resolve_cs_weighting_raises_when_missing(self, factor_model):
        with pytest.raises(ValueError, match="benchmark_weights"):
            factor_model._resolve_cs_weighting(CSWeighting.BENCHMARK, latest=False)
        with pytest.raises(ValueError, match="regression_weights"):
            factor_model._resolve_cs_weighting(CSWeighting.REGRESSION, latest=False)

    def test_resolve_cs_weighting_fallback_warns(self, factor_model):
        with pytest.warns(UserWarning, match="Falling back"):
            weights = factor_model._resolve_cs_weighting(
                CSWeighting.BENCHMARK,
                latest=False,
                fallback_cs_weighting=CSWeighting.IDENTITY,
            )
        assert weights is None

    def test_correlation_weighted_vs_equal(self, factor_model_with_weights):
        fm = factor_model_with_weights
        fig_bmk = fm.plot_exposure_correlation(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )
        fig_eq = fm.plot_exposure_correlation(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        z_bmk = np.array(fig_bmk.data[0].z)
        z_eq = np.array(fig_eq.data[0].z)
        assert z_bmk.shape == z_eq.shape
        assert not np.allclose(z_bmk, z_eq, atol=1e-6), (
            "weighted and equal-weighted correlation should differ"
        )
        np.testing.assert_allclose(np.diag(z_bmk), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.diag(z_eq), 1.0, atol=1e-10)

    def test_correlation_symmetric(self, factor_model_with_weights):
        fm = factor_model_with_weights
        fig = fm.plot_exposure_correlation(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )
        z = np.array(fig.data[0].z)
        np.testing.assert_allclose(z, z.T, atol=1e-10)

    def test_dispersion_weighted_vs_equal(self, factor_model_with_weights):
        fm = factor_model_with_weights
        fig_bmk = fm.plot_exposure_dispersion(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )
        fig_eq = fm.plot_exposure_dispersion(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        y_bmk = np.array(fig_bmk.data[0].y, dtype=float)
        y_eq = np.array(fig_eq.data[0].y, dtype=float)
        assert not np.allclose(y_bmk, y_eq, atol=1e-6), (
            "weighted and equal-weighted dispersion should differ"
        )
        assert (y_bmk > 0).all()
        assert (y_eq > 0).all()

    def test_stability_weighted_vs_equal(self, factor_model_with_weights):
        fm = factor_model_with_weights
        fig_bmk = fm.plot_exposure_stability(
            families=None, cs_weighting=CSWeighting.BENCHMARK
        )
        fig_eq = fm.plot_exposure_stability(
            families=None, cs_weighting=CSWeighting.IDENTITY
        )
        y_bmk = np.array(fig_bmk.data[0].y, dtype=float)
        y_eq = np.array(fig_eq.data[0].y, dtype=float)
        assert not np.allclose(y_bmk, y_eq, atol=1e-6), (
            "weighted and equal-weighted stability should differ"
        )

    def test_stability_regression_weighting(self, factor_model_with_weights):
        fm = factor_model_with_weights
        fig = fm.plot_exposure_stability(
            families=None, cs_weighting=CSWeighting.REGRESSION
        )
        assert isinstance(fig, go.Figure)

    def test_summary_stability_weighting(self, factor_model_with_weights):
        fm = factor_model_with_weights
        df_bmk = fm.summary(stability_cs_weighting=CSWeighting.BENCHMARK)
        df_eq = fm.summary(stability_cs_weighting=CSWeighting.IDENTITY)
        assert not np.allclose(
            df_bmk["stability"].values,
            df_eq["stability"].values,
            atol=1e-6,
        )

    def test_select_observations_preserves_benchmark_weights(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        target = fm.observations[5:15]
        aligned = fm.select_observations(target)
        np.testing.assert_array_equal(
            aligned.benchmark_weights, fm.benchmark_weights[5:15]
        )


class TestSelectAssets:
    def test_identity_returns_self(self, factor_model):
        assert factor_model.select_assets() is factor_model
        mask = np.ones(len(factor_model.asset_names), dtype=bool)
        assert factor_model.select_assets(mask) is factor_model

    def test_boolean_mask_slices_asset_fields(self, factor_model_with_weights):
        fm = factor_model_with_weights
        idio_mu = np.arange(len(fm.asset_names), dtype=float)
        fm = replace(fm, idio_mu=idio_mu)
        mask = np.zeros(len(fm.asset_names), dtype=bool)
        mask[[0, 2, 5, 7]] = True

        subset = fm.select_assets(mask)

        np.testing.assert_array_equal(subset.asset_names, fm.asset_names[mask])
        np.testing.assert_array_equal(subset.loading_matrix, fm.loading_matrix[mask])
        np.testing.assert_array_equal(subset.exposures, fm.exposures[:, mask, :])
        np.testing.assert_array_equal(subset.idio_covariance, fm.idio_covariance[mask])
        np.testing.assert_array_equal(subset.idio_mu, idio_mu[mask])
        np.testing.assert_array_equal(subset.idio_returns, fm.idio_returns[:, mask])
        np.testing.assert_array_equal(subset.idio_variances, fm.idio_variances[:, mask])
        np.testing.assert_array_equal(
            subset.regression_weights, fm.regression_weights[:, mask]
        )
        np.testing.assert_array_equal(
            subset.benchmark_weights, fm.benchmark_weights[:, mask]
        )
        assert subset.factor_returns is fm.factor_returns
        assert subset.family_constraint_basis is fm.family_constraint_basis

    def test_label_selector_slices_assets(self, factor_model):
        assets = factor_model.asset_names[[1, 4, 8]]
        subset = factor_model.select_assets(assets)
        np.testing.assert_array_equal(subset.asset_names, assets)
        np.testing.assert_array_equal(
            subset.loading_matrix, factor_model.loading_matrix[[1, 4, 8]]
        )

    def test_integer_selector_can_reorder_assets(self, factor_model):
        indices = np.array([5, 2, 0])
        subset = factor_model.select_assets(indices)
        np.testing.assert_array_equal(
            subset.asset_names, factor_model.asset_names[indices]
        )
        np.testing.assert_array_equal(
            subset.loading_matrix, factor_model.loading_matrix[indices]
        )

    def test_full_idio_covariance_sliced_on_both_axes(self, factor_model):
        full_idio_covariance = np.arange(
            len(factor_model.asset_names) ** 2, dtype=float
        ).reshape(len(factor_model.asset_names), len(factor_model.asset_names))
        fm = replace(factor_model, idio_covariance=full_idio_covariance)
        indices = np.array([3, 1, 5])
        subset = fm.select_assets(indices)
        np.testing.assert_array_equal(
            subset.idio_covariance, full_idio_covariance[np.ix_(indices, indices)]
        )

    def test_duplicate_assets_raise(self, factor_model):
        assets = factor_model.asset_names[[1, 1, 3]]
        with pytest.raises(ValueError, match="duplicate-free"):
            factor_model.select_assets(assets)

    def test_out_of_bounds_assets_raise(self, factor_model):
        with pytest.raises(ValueError, match="out-of-bounds"):
            factor_model.select_assets([0, len(factor_model.asset_names)])

    def test_slim_all_assets_preserves_static_fields(self, factor_model_with_weights):
        fm = factor_model_with_weights
        subset = fm.select_assets(slim=True)
        assert subset.asset_names is fm.asset_names
        assert subset.loading_matrix is fm.loading_matrix
        assert subset.idio_covariance is fm.idio_covariance
        assert subset.regression_weights is fm.regression_weights
        assert subset.exposures is None
        assert subset.idio_returns is None
        assert subset.idio_variances is None
        assert subset.benchmark_weights is None

    def test_return_distribution_slim_uses_all_assets_fast_path(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        distribution = ReturnDistribution(
            mu=np.ones(len(fm.asset_names)),
            covariance=np.eye(len(fm.asset_names)),
            returns=np.ones((len(fm.observations), len(fm.asset_names))),
            factor_model=fm,
        )

        subset = distribution.investable_subset(slim=True)

        assert subset.factor_model is not None
        assert subset.factor_model.asset_names is fm.asset_names
        assert subset.factor_model.loading_matrix is fm.loading_matrix
        assert subset.factor_model.exposures is None
        assert subset.factor_model.benchmark_weights is None


class TestSelectObservations:
    def test_identity_returns_self(self, factor_model):
        aligned = factor_model.select_observations(factor_model.observations)
        assert aligned is factor_model

    def test_contiguous_slice_is_view(self, factor_model):
        target = factor_model.observations[5:15]
        aligned = factor_model.select_observations(target)
        assert len(aligned.observations) == 10
        assert np.shares_memory(aligned.factor_returns, factor_model.factor_returns)

    def test_non_contiguous_slice(self, factor_model):
        target = factor_model.observations[np.array([0, 5, 10, 20])]
        aligned = factor_model.select_observations(target)
        assert len(aligned.observations) == 4
        np.testing.assert_array_equal(aligned.observations, target)

    def test_positional_slice(self, factor_model):
        aligned = factor_model.select_observations(slice(5, 15))
        np.testing.assert_array_equal(
            aligned.observations, factor_model.observations[5:15]
        )

    def test_boolean_mask(self, factor_model):
        mask = np.zeros(len(factor_model.observations), dtype=bool)
        mask[[0, 3, 5]] = True
        aligned = factor_model.select_observations(mask)
        np.testing.assert_array_equal(
            aligned.observations, factor_model.observations[mask]
        )

    def test_static_fields_shared(self, factor_model):
        target = factor_model.observations[5:15]
        aligned = factor_model.select_observations(target)
        assert aligned.loading_matrix is factor_model.loading_matrix
        assert aligned.factor_covariance is factor_model.factor_covariance
        assert aligned.idio_covariance is factor_model.idio_covariance
        assert aligned.factor_mu is factor_model.factor_mu
        assert aligned.factor_families is factor_model.factor_families
        assert aligned.asset_names is factor_model.asset_names
        assert aligned.factor_names is factor_model.factor_names

    def test_time_varying_fields_sliced(self, factor_model):
        target = factor_model.observations[10:20]
        aligned = factor_model.select_observations(target)
        np.testing.assert_array_equal(
            aligned.factor_returns, factor_model.factor_returns[10:20]
        )
        np.testing.assert_array_equal(aligned.exposures, factor_model.exposures[10:20])
        np.testing.assert_array_equal(
            aligned.idio_returns, factor_model.idio_returns[10:20]
        )

    def test_none_fields_stay_none(self, factor_model_no_ts):
        target = factor_model_no_ts.observations[5:15]
        aligned = factor_model_no_ts.select_observations(target)
        assert aligned.factor_returns is None
        assert aligned.exposures is None
        assert aligned.idio_returns is None

    def test_missing_observations_raises(self, factor_model):
        bad_obs = np.array(["2099-01-01", "2099-02-01"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="not found in FactorModel"):
            factor_model.select_observations(bad_obs)

    def test_reordered_observations_raise(self, factor_model):
        target = factor_model.observations[np.array([5, 2, 10])]
        with pytest.raises(ValueError, match="same relative order"):
            factor_model.select_observations(target)

    def test_duplicate_observations_raise(self, factor_model):
        target = factor_model.observations[np.array([3, 3, 8])]
        with pytest.raises(ValueError, match="duplicate-free subset"):
            factor_model.select_observations(target)

    def test_exposure_lag_preserved(self, factor_model):
        target = factor_model.observations[:10]
        aligned = factor_model.select_observations(target)
        assert aligned.exposure_lag == factor_model.exposure_lag

    def test_loading_matrix_not_updated(self, factor_model):
        target = factor_model.observations[:10]
        aligned = factor_model.select_observations(target)
        np.testing.assert_array_equal(
            aligned.loading_matrix, factor_model.loading_matrix
        )


# ------------------------------------------------------------------
# CS regression scores: cheap (R2, adjusted R2, AIC, BIC)
# ------------------------------------------------------------------
class TestCSRegressionScores:
    def test_returns_dataframe(self, factor_model):
        df = factor_model.cs_regression_scores
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["r2", "adjusted_r2", "aic", "bic"]

    def test_shape(self, factor_model):
        df = factor_model.cs_regression_scores
        n_aligned = factor_model.factor_returns.shape[0] - factor_model.exposure_lag
        assert len(df) == n_aligned

    def test_r2_in_range(self, factor_model):
        valid = factor_model.cs_regression_scores["r2"].dropna()
        assert (valid <= 1.0 + 1e-10).all()

    def test_adjusted_r2_le_r2(self, factor_model):
        df = factor_model.cs_regression_scores
        r2 = df["r2"].values
        adj = df["adjusted_r2"].values
        valid = np.isfinite(r2) & np.isfinite(adj)
        assert (adj[valid] <= r2[valid] + 1e-12).all()

    def test_bic_penalizes_more_than_aic(self, factor_model):
        df = factor_model.cs_regression_scores
        aic = df["aic"].values
        bic = df["bic"].values
        n_assets = factor_model.asset_names.shape[0]
        n_factors = factor_model.factor_names.shape[0]
        if n_assets > np.exp(2) and n_factors > 0:
            valid = np.isfinite(aic) & np.isfinite(bic)
            assert (bic[valid] >= aic[valid] - 1e-10).all()

    def test_information_criteria_invariant_to_weight_scale(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        scaled = replace(fm, regression_weights=fm.regression_weights * 17.0)

        pd.testing.assert_frame_equal(
            fm.cs_regression_scores,
            scaled.cs_regression_scores,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_information_criteria_are_negative_infinity_for_exact_fit(
        self, factor_model
    ):
        fm = replace(
            factor_model, idio_returns=np.zeros_like(factor_model.idio_returns)
        )
        df = fm.cs_regression_scores

        assert (df["r2"] == 1.0).all()
        assert (df["adjusted_r2"] == 1.0).all()
        assert np.isneginf(df["aic"]).all()
        assert np.isneginf(df["bic"]).all()

    def test_requires_time_series(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            _ = factor_model_no_ts.cs_regression_scores


# ------------------------------------------------------------------
# CS regression and exposure design diagnostics
# ------------------------------------------------------------------
class TestCSRegressionTStats:
    def test_returns_dataframe(self, factor_model):
        df = factor_model.cs_regression_t_stats
        assert isinstance(df, pd.DataFrame)

    def test_shape(self, factor_model):
        df = factor_model.cs_regression_t_stats
        n_aligned = factor_model.factor_returns.shape[0] - factor_model.exposure_lag
        assert df.shape == (n_aligned, len(factor_model.factor_names))

    def test_columns_are_factor_names(self, factor_model):
        df = factor_model.cs_regression_t_stats
        assert list(df.columns) == list(factor_model.factor_names)

    def test_cached(self, factor_model):
        _ = factor_model.cs_regression_t_stats
        assert "_gram_diagnostics" in factor_model.__dict__
        _ = factor_model.exposure_vif
        _ = factor_model.exposure_condition_number

    def test_requires_time_series(self, factor_model_no_ts):
        with pytest.raises(ValueError, match="exposures"):
            _ = factor_model_no_ts.cs_regression_t_stats


class TestExposureVIF:
    def test_returns_dataframe(self, factor_model):
        df = factor_model.exposure_vif
        assert isinstance(df, pd.DataFrame)

    def test_values_positive(self, factor_model):
        df = factor_model.exposure_vif
        valid = df.values[np.isfinite(df.values)]
        assert (valid > 0).all()


class TestExposureConditionNumber:
    def test_returns_series(self, factor_model):
        s = factor_model.exposure_condition_number
        assert isinstance(s, pd.Series)
        assert s.name == "exposure_condition_number"

    def test_values_ge_one(self, factor_model):
        s = factor_model.exposure_condition_number
        valid = s.dropna()
        assert (valid >= 1.0 - 1e-10).all()


class TestCSRegressionTStatExceedanceRate:
    def test_returns_series(self, factor_model):
        hr = factor_model.cs_regression_t_stat_exceedance_rate()
        assert isinstance(hr, pd.Series)
        assert hr.name == "cs_regression_t_stat_exceedance_rate"
        assert len(hr) == len(factor_model.factor_names)

    def test_in_unit_interval(self, factor_model):
        hr = factor_model.cs_regression_t_stat_exceedance_rate()
        assert (hr >= 0).all() and (hr <= 1).all()

    def test_custom_threshold(self, factor_model):
        hr_low = factor_model.cs_regression_t_stat_exceedance_rate(threshold=0.5)
        hr_high = factor_model.cs_regression_t_stat_exceedance_rate(threshold=5.0)
        assert (hr_low >= hr_high - 1e-10).all()

    def test_constant_exposure_factor(self):
        """Hit rate must stay in [0, 1] even when a factor has zero SE.

        A constant exposure column (e.g. global intercept or industry
        dummy) produces a singular Gram sub-matrix for that factor,
        yielding SE = 0 and undefined t-statistics (NaN).  Previously
        this produced inf t-stats, causing an exceedance rate above 1.
        """
        rng = np.random.default_rng(99)
        n_obs, n_assets, n_factors = 20, 40, 3
        factor_names = np.array(["style", "global", "industry"])
        asset_names = np.array([f"a{i}" for i in range(n_assets)])
        observations = np.arange(n_obs)

        exposures = rng.standard_normal((n_obs, n_assets, n_factors))
        exposures[:, :, 1] = 1.0  # constant global factor
        exposures[:, :20, 2] = 1.0  # industry dummy
        exposures[:, 20:, 2] = 0.0

        factor_returns = rng.standard_normal((n_obs, n_factors)) * 0.01
        idio_returns = rng.standard_normal((n_obs, n_assets)) * 0.05

        A = rng.standard_normal((n_factors, n_factors))
        factor_cov = A @ A.T / n_factors

        fm = FactorModel(
            observations=observations,
            asset_names=asset_names,
            factor_names=factor_names,
            factor_families=None,
            loading_matrix=exposures[-1],
            exposures=exposures,
            factor_covariance=factor_cov,
            factor_mu=np.zeros(n_factors),
            factor_returns=factor_returns,
            idio_covariance=np.ones(n_assets) * 0.01,
            idio_mu=None,
            idio_returns=idio_returns,
            idio_variances=None,
            exposure_lag=0,
        )

        hr = fm.cs_regression_t_stat_exceedance_rate(threshold=2.0)
        assert (hr >= 0).all(), (
            f"CS regression t-stat exceedance rate has negative values: {hr.values}"
        )
        assert (hr <= 1).all(), (
            f"CS regression t-stat exceedance rate exceeds 1.0: {hr.values}"
        )

        t = fm.cs_regression_t_stats
        assert np.all(np.isfinite(t.values) | np.isnan(t.values)), (
            "CS regression t-stats must not contain inf"
        )


# ------------------------------------------------------------------
# CS regression and exposure diagnostic plots
# ------------------------------------------------------------------
class TestPlotCSRegressionScores:
    @pytest.mark.parametrize("score", ["r2", "adjusted_r2", "aic", "bic"])
    def test_returns_figure(self, factor_model, score):
        fig = factor_model.plot_cs_regression_scores(score=score, window=5)
        assert isinstance(fig, go.Figure)

    def test_default_score(self, factor_model):
        fig = factor_model.plot_cs_regression_scores(window=5)
        assert "Adjusted R\u00b2" in fig.layout.title.text

    def test_custom_title(self, factor_model):
        fig = factor_model.plot_cs_regression_scores(title="My Title")
        assert fig.layout.title.text == "My Title"

    def test_two_traces(self, factor_model):
        fig = factor_model.plot_cs_regression_scores(window=5)
        assert len(fig.data) == 2

    def test_invalid_score(self, factor_model):
        with pytest.raises(ValueError, match="score"):
            factor_model.plot_cs_regression_scores(score="bad")


class TestPlotCSRegressionTStats:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats()
        assert isinstance(fig, go.Figure)

    def test_all_factors_shown(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats()
        assert len(fig.data) == len(factor_model.factor_names)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats(factors=["factor_0", "factor_1"])
        assert len(fig.data) == 2

    def test_rolling_window(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats(window=5)
        assert isinstance(fig, go.Figure)
        assert "Rolling" in fig.layout.title.text

    def test_custom_title(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats(title="Custom")
        assert fig.layout.title.text == "Custom"

    def test_plot_labels_formatted(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stats(factors=["factor_0"])
        assert fig.data[0].name == "Factor 0"


class TestPlotCSRegressionTStatExceedanceRate:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stat_exceedance_rate()
        assert isinstance(fig, go.Figure)

    def test_bar_count_matches_factors(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stat_exceedance_rate()
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == len(factor_model.factor_names)

    def test_custom_threshold(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stat_exceedance_rate(threshold=1.0)
        assert "1.0" in fig.layout.title.text

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_cs_regression_t_stat_exceedance_rate(
            factors=["factor_0", "factor_1"]
        )
        assert len(fig.data[0].x) == 2

    def test_families_filter(self, factor_model_with_families):
        fig = factor_model_with_families.plot_cs_regression_t_stat_exceedance_rate(
            families="style"
        )
        assert len(fig.data[0].x) == 3


class TestPlotExposureVIF:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_exposure_vif()
        assert isinstance(fig, go.Figure)

    def test_all_factors_shown(self, factor_model):
        fig = factor_model.plot_exposure_vif()
        assert len(fig.data) == len(factor_model.factor_names)

    def test_factor_subset(self, factor_model):
        fig = factor_model.plot_exposure_vif(factors=["factor_0", "factor_1"])
        assert len(fig.data) == 2

    def test_rolling_window(self, factor_model):
        fig = factor_model.plot_exposure_vif(window=5)
        assert isinstance(fig, go.Figure)
        assert "Rolling" in fig.layout.title.text


class TestPlotExposureConditionNumber:
    def test_returns_figure(self, factor_model):
        fig = factor_model.plot_exposure_condition_number(window=5)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, factor_model):
        fig = factor_model.plot_exposure_condition_number(title="Custom")
        assert fig.layout.title.text == "Custom"


class TestScoreCaching:
    def test_gram_diagnostics_cached_together(self, factor_model):
        _ = factor_model.cs_regression_t_stats
        cached = factor_model.__dict__.get("_gram_diagnostics")
        assert cached is not None
        assert cached.t_stats is not None
        assert cached.vif is not None
        assert cached.condition_number is not None

    def test_select_observations_does_not_share_cache(self, factor_model):
        _ = factor_model.cs_regression_t_stats
        target = factor_model.observations[5:15]
        aligned = factor_model.select_observations(target)
        assert "_gram_diagnostics" not in aligned.__dict__


class TestGramDiagnosticsWithBasis:
    """Gram diagnostics and plots when a basket-neutral basis is present."""

    def test_cs_regression_t_stats_shape(self, factor_model_with_basis):
        fm = factor_model_with_basis
        t = fm.cs_regression_t_stats
        K_red = fm.family_constraint_basis.n_reduced_factors
        assert t.shape[1] == K_red

    def test_cs_regression_t_stats_columns_match_gram_names(
        self, factor_model_with_basis
    ):
        fm = factor_model_with_basis
        t = fm.cs_regression_t_stats
        assert list(t.columns) == list(fm._reduced_factor_names)

    def test_vif_shape(self, factor_model_with_basis):
        fm = factor_model_with_basis
        v = fm.exposure_vif
        K_red = fm.family_constraint_basis.n_reduced_factors
        assert v.shape[1] == K_red

    def test_exposure_condition_number_shape(self, factor_model_with_basis):
        fm = factor_model_with_basis
        c = fm.exposure_condition_number
        assert c.shape == (fm.factor_returns.shape[0] - fm.exposure_lag,)

    def test_exceedance_rate_length(self, factor_model_with_basis):
        fm = factor_model_with_basis
        hr = fm.cs_regression_t_stat_exceedance_rate()
        K_red = fm.family_constraint_basis.n_reduced_factors
        assert hr.shape == (K_red,)

    def test_adjusted_r2_uses_n_regressors(self, factor_model_with_basis):
        fm = factor_model_with_basis
        ar2 = fm.cs_regression_scores["adjusted_r2"]
        assert ar2.shape[0] == fm.factor_returns.shape[0] - fm.exposure_lag

    def test_plot_cs_regression_t_stats_all(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_cs_regression_t_stats()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == fm.family_constraint_basis.n_reduced_factors

    def test_plot_cs_regression_t_stats_factor_subset(self, factor_model_with_basis):
        fm = factor_model_with_basis
        gram_names = list(fm._reduced_factor_names)
        fig = fm.plot_cs_regression_t_stats(factors=gram_names[:2])
        assert len(fig.data) == 2

    def test_plot_cs_regression_t_stats_family_filter(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_cs_regression_t_stats(families="industry")
        dropped = fm.family_constraint_basis.dropped_full_indices
        n_industry_full = sum(1 for f in fm.factor_families if f == "industry")
        assert len(fig.data) == n_industry_full - len(dropped)

    def test_plot_cs_regression_t_stats_rolling(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_cs_regression_t_stats(window=5)
        assert "Rolling" in fig.layout.title.text

    def test_plot_exposure_vif_all(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_exposure_vif()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == fm.family_constraint_basis.n_reduced_factors

    def test_plot_exposure_vif_family_filter(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_exposure_vif(families="industry")
        dropped = fm.family_constraint_basis.dropped_full_indices
        n_industry_full = sum(1 for f in fm.factor_families if f == "industry")
        assert len(fig.data) == n_industry_full - len(dropped)

    def test_plot_cs_regression_t_stat_exceedance_rate(self, factor_model_with_basis):
        fm = factor_model_with_basis
        fig = fm.plot_cs_regression_t_stat_exceedance_rate()
        assert isinstance(fig, go.Figure)
        K_red = fm.family_constraint_basis.n_reduced_factors
        assert len(fig.data[0].x) == K_red

    def test_dropped_factor_not_in_gram_names(self, factor_model_with_basis):
        fm = factor_model_with_basis
        gram_names = set(fm._reduced_factor_names)
        for idx in fm.family_constraint_basis.dropped_full_indices:
            assert fm.factor_names[idx] not in gram_names


class TestFactorModelMissingDataContract:
    def test_gram_diagnostics_exclude_nonfinite_exposures(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        exposures = fm.exposures.copy()
        exposures[0, 0, 0] = np.nan
        fm_missing = replace(fm, exposures=exposures)

        t_stats = fm_missing.cs_regression_t_stats

        assert not np.isnan(t_stats.iloc[0].to_numpy()).all()
        assert np.isfinite(t_stats.iloc[1].to_numpy()).any()

    def test_r2_excludes_partial_missing_idio_returns(self, factor_model_with_weights):
        fm = factor_model_with_weights
        idio_returns = fm.idio_returns.copy()
        idio_returns[1, 0] = np.nan
        fm_missing = replace(fm, idio_returns=idio_returns)

        r2 = fm_missing.cs_regression_scores["r2"]

        assert np.isfinite(r2.iloc[0])

    def test_r2_is_nan_when_factor_returns_missing_for_observation(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        factor_returns = fm.factor_returns.copy()
        factor_returns[1, :] = np.nan
        fm_missing = replace(fm, factor_returns=factor_returns)

        r2 = fm_missing.cs_regression_scores["r2"]

        assert np.isnan(r2.iloc[0])
        assert np.isfinite(r2.iloc[1:].to_numpy()).any()

    def test_exposure_diagnostics_ignore_missing_factor_returns(
        self, factor_model_with_weights
    ):
        fm = factor_model_with_weights
        factor_returns = fm.factor_returns.copy()
        factor_returns[1, :] = np.nan
        fm_missing = replace(fm, factor_returns=factor_returns)

        t_stats = fm_missing.cs_regression_t_stats
        vif = fm_missing.exposure_vif
        condition_number = fm_missing.exposure_condition_number

        assert np.isnan(t_stats.iloc[0].to_numpy()).all()
        assert np.isfinite(vif.iloc[0].to_numpy()).all()
        assert np.isfinite(condition_number.iloc[0])


# ------------------------------------------------------------------
# Attribution methods
# ------------------------------------------------------------------
class TestFactorModelAttribution:
    """Tests for FactorModel attribution convenience methods."""

    @pytest.fixture()
    def fm_with_weights(self):
        """Build a factor model and matching weights/returns for attribution."""
        rng = np.random.default_rng(42)
        base_fm = _make_factor_model(n_obs=100, n_assets=10, n_factors=3, seed=42)
        regression_weights = rng.uniform(
            0.1,
            1.0,
            size=(base_fm.observations.shape[0], base_fm.asset_names.shape[0]),
        )
        fm = FactorModel(
            observations=base_fm.observations,
            asset_names=base_fm.asset_names,
            factor_names=base_fm.factor_names,
            factor_families=base_fm.factor_families,
            loading_matrix=base_fm.loading_matrix,
            exposures=base_fm.exposures,
            factor_covariance=base_fm.factor_covariance,
            factor_mu=base_fm.factor_mu,
            factor_returns=base_fm.factor_returns,
            idio_covariance=base_fm.idio_covariance,
            idio_mu=base_fm.idio_mu,
            idio_returns=base_fm.idio_returns,
            idio_variances=base_fm.idio_variances,
            exposure_lag=base_fm.exposure_lag,
            regression_weights=regression_weights,
            benchmark_weights=base_fm.benchmark_weights,
            family_constraint_basis=base_fm.family_constraint_basis,
        )

        w = np.abs(rng.standard_normal(10)) + 0.1
        w /= w.sum()

        rets = rng.standard_normal(100) * 0.01
        return fm, w, rets

    # --- predicted_attribution ---

    def test_predicted_attribution_returns_attribution(self, fm_with_weights):
        from skfolio.attribution import Attribution

        fm, w, _ = fm_with_weights
        result = fm.predicted_attribution(weights=w)
        assert isinstance(result, Attribution)

    def test_predicted_attribution_decomposition(self, fm_with_weights):
        fm, w, _ = fm_with_weights
        result = fm.predicted_attribution(weights=w, annualization_factor=1)
        np.testing.assert_almost_equal(
            result.systematic.vol_contrib + result.idio.vol_contrib,
            result.total.vol,
            decimal=10,
        )

    def test_predicted_attribution_no_ts_works(self):
        fm = _make_factor_model(with_time_series=False)
        rng = np.random.default_rng(0)
        w = np.abs(rng.standard_normal(30)) + 0.1
        w /= w.sum()
        result = fm.predicted_attribution(weights=w)
        assert result.total.vol > 0

    # --- realized_attribution ---

    def test_realized_attribution_returns_attribution(self, fm_with_weights):
        from skfolio.attribution import Attribution

        fm, w, rets = fm_with_weights
        result = fm.realized_attribution(weights=w, portfolio_returns=rets)
        assert isinstance(result, Attribution)

    def test_realized_attribution_decomposition(self, fm_with_weights):
        fm, w, rets = fm_with_weights
        result = fm.realized_attribution(
            weights=w, portfolio_returns=rets, annualization_factor=1
        )
        sum_vol = (
            np.sum(result.factors.vol_contrib)
            + result.idio.vol_contrib
            + result.unexplained.vol_contrib
        )
        np.testing.assert_almost_equal(sum_vol, result.total.vol, decimal=10)

    def test_realized_attribution_requires_ts(self):
        fm = _make_factor_model(with_time_series=False)
        rng = np.random.default_rng(0)
        w = np.abs(rng.standard_normal(30)) + 0.1
        w /= w.sum()
        with pytest.raises(ValueError, match="factor_returns"):
            fm.realized_attribution(weights=w, portfolio_returns=np.zeros(50))

    # --- rolling_realized_attribution ---

    def test_rolling_realized_returns_attribution(self, fm_with_weights):
        from skfolio.attribution import Attribution

        fm, w, rets = fm_with_weights
        result = fm.rolling_realized_attribution(
            weights=w,
            portfolio_returns=rets,
            window_size=30,
            step=10,
        )
        assert isinstance(result, Attribution)
        assert result.is_rolling is True

    def test_rolling_realized_window_count(self, fm_with_weights):
        fm, w, rets = fm_with_weights
        result = fm.rolling_realized_attribution(
            weights=w,
            portfolio_returns=rets,
            window_size=30,
            step=10,
        )
        assert len(result.observations) > 0
        assert result.total.vol.shape[0] == len(result.observations)

    def test_rolling_realized_decomposition_per_window(self, fm_with_weights):
        fm, w, rets = fm_with_weights
        result = fm.rolling_realized_attribution(
            weights=w,
            portfolio_returns=rets,
            window_size=30,
            step=20,
            annualization_factor=1,
        )
        for i in range(len(result.observations)):
            sum_vol = (
                np.sum(result.factors.vol_contrib[i])
                + result.idio.vol_contrib[i]
                + result.unexplained.vol_contrib[i]
            )
            np.testing.assert_almost_equal(sum_vol, result.total.vol[i], decimal=10)

    def test_rolling_realized_requires_ts(self):
        fm = _make_factor_model(with_time_series=False)
        rng = np.random.default_rng(0)
        w = np.abs(rng.standard_normal(30)) + 0.1
        w /= w.sum()
        with pytest.raises(ValueError, match="factor_returns"):
            fm.rolling_realized_attribution(
                weights=w,
                portfolio_returns=np.zeros(50),
                window_size=20,
                step=10,
            )

    def test_rolling_realized_window_too_large_raises(self, fm_with_weights):
        fm, w, rets = fm_with_weights
        with pytest.raises(ValueError, match="exceeds"):
            fm.rolling_realized_attribution(
                weights=w,
                portfolio_returns=rets,
                window_size=500,
                step=10,
            )


# ---------------------------------------------------------------------------
# Helpers for orthogonal property tests
# ---------------------------------------------------------------------------
def _make_minimal_factor_model(
    loading,
    idio_covariance,
    regression_weights=None,
    family_constraint_basis=None,
    factor_covariance=None,
):
    """Build a minimal FactorModel for property tests."""
    n_assets, n_factors = loading.shape
    if factor_covariance is None:
        factor_covariance = np.eye(n_factors)
    return FactorModel(
        observations=np.arange(1),
        asset_names=np.array([f"A{i}" for i in range(n_assets)]),
        factor_names=np.array([f"F{j}" for j in range(n_factors)]),
        factor_families=None,
        loading_matrix=loading,
        exposures=None,
        factor_covariance=factor_covariance,
        factor_mu=np.zeros(n_factors),
        factor_returns=None,
        idio_covariance=idio_covariance,
        idio_mu=None,
        idio_returns=None,
        idio_variances=None,
        regression_weights=regression_weights,
        family_constraint_basis=family_constraint_basis,
    )


# ---------------------------------------------------------------------------
# FactorModel.effective_loading_matrix
# ---------------------------------------------------------------------------
class TestEffectiveLoadingMatrix:
    def test_passthrough_when_no_basis(self):
        loading = np.eye(5, 3)
        fm = _make_minimal_factor_model(loading, np.ones(5) * 0.01)
        np.testing.assert_array_equal(fm.effective_loading_matrix, loading)

    def test_reduces_rank_with_family_constraint_basis(self):
        n_assets, n_factors = 6, 4
        rng = np.random.default_rng(50)
        loading = rng.standard_normal((n_assets, n_factors))

        constrained_family = FamilyConstraint(
            family_name="industry",
            full_factor_indices=np.array([0, 1]),
            dropped_index_in_family=0,
        )
        basis = FamilyConstraintBasis(
            n_full_factors=n_factors,
            family_constraints=(constrained_family,),
            constraint_ratios=rng.uniform(0.3, 0.7, size=(1, 1)),
        )

        fm = _make_minimal_factor_model(
            loading,
            np.ones(n_assets) * 0.01,
            family_constraint_basis=basis,
        )
        reduced = fm.effective_loading_matrix
        assert reduced.shape == (n_assets, n_factors - 1)


class TestEffectiveFactorCovariance:
    def test_passthrough_when_no_basis(self):
        loading = np.eye(5, 3)
        matrix = np.array(
            [
                [2.0, 0.2, 0.1],
                [0.2, 1.5, 0.3],
                [0.1, 0.3, 1.0],
            ]
        )
        factor_covariance = matrix @ matrix.T
        fm = _make_minimal_factor_model(
            loading,
            np.ones(5) * 0.01,
            factor_covariance=factor_covariance,
        )

        np.testing.assert_array_equal(fm.effective_factor_covariance, factor_covariance)

    def test_reduces_to_full_rank_retained_block(self):
        n_assets, n_factors = 6, 4
        rng = np.random.default_rng(51)
        loading = rng.standard_normal((n_assets, n_factors))
        constrained_family = FamilyConstraint(
            family_name="industry",
            full_factor_indices=np.array([0, 1]),
            dropped_index_in_family=0,
        )
        basis = FamilyConstraintBasis(
            n_full_factors=n_factors,
            family_constraints=(constrained_family,),
            constraint_ratios=rng.uniform(0.3, 0.7, size=(1, 1)),
        )
        matrix = rng.standard_normal((basis.n_reduced_factors, basis.n_reduced_factors))
        covariance_reduced = matrix @ matrix.T / basis.n_reduced_factors
        covariance_full = basis.expand_factor_covariance(covariance_reduced)
        fm = _make_minimal_factor_model(
            loading,
            np.ones(n_assets) * 0.01,
            family_constraint_basis=basis,
            factor_covariance=covariance_full,
        )

        effective = fm.effective_factor_covariance

        assert effective.shape == (basis.n_reduced_factors, basis.n_reduced_factors)
        np.testing.assert_allclose(
            effective,
            covariance_full[
                np.ix_(basis.retained_full_indices, basis.retained_full_indices)
            ],
            atol=1e-12,
        )
        assert np.linalg.eigvalsh(effective).min() > 0

    def test_systematic_covariance_matches_full_basis(self):
        n_assets, n_factors = 6, 4
        rng = np.random.default_rng(52)
        loading = rng.standard_normal((n_assets, n_factors))
        constrained_family = FamilyConstraint(
            family_name="industry",
            full_factor_indices=np.array([0, 1]),
            dropped_index_in_family=0,
        )
        basis = FamilyConstraintBasis(
            n_full_factors=n_factors,
            family_constraints=(constrained_family,),
            constraint_ratios=rng.uniform(0.3, 0.7, size=(1, 1)),
        )
        matrix = rng.standard_normal((basis.n_reduced_factors, basis.n_reduced_factors))
        covariance_reduced = matrix @ matrix.T / basis.n_reduced_factors
        covariance_full = basis.expand_factor_covariance(covariance_reduced)
        fm = _make_minimal_factor_model(
            loading,
            np.ones(n_assets) * 0.01,
            family_constraint_basis=basis,
            factor_covariance=covariance_full,
        )

        systematic_full = loading @ covariance_full @ loading.T
        systematic_effective = (
            fm.effective_loading_matrix
            @ fm.effective_factor_covariance
            @ fm.effective_loading_matrix.T
        )

        np.testing.assert_allclose(systematic_effective, systematic_full, atol=1e-12)
