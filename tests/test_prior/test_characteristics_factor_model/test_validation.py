"""Validation and error-handling tests for CharacteristicsFactorModel internals.

Tests in this file verify input validation, name resolution, and error
messages -- they do *not* rely on statistical recovery from a DGP.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from skfolio._constants import (
    _BENCHMARK_WEIGHTS,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _REGRESSION_WEIGHTS,
)
from skfolio.exceptions import DuplicateGroupsError
from skfolio.factor_exposure import DerivedFactor
from skfolio.linear_model import CSLinearRegression
from skfolio.moments import EWCovariance
from skfolio.moments.variance import EWVariance
from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior, ReturnDistribution
from skfolio.utils._factor_tools import (
    _neutralize_exposures,
    _resolve_factor_name,
)

from .conftest import make_panel, passthrough_factor


class WeightedEmpiricalPrior(EmpiricalPrior):
    """Empirical prior that exposes deterministic sample weights for tests."""

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        dist = self.return_distribution_
        sample_weight = np.arange(1, dist.returns.shape[0] + 1, dtype=float)
        self.return_distribution_ = ReturnDistribution(
            mu=dist.mu,
            covariance=dist.covariance,
            returns=dist.returns,
            sample_weight=sample_weight,
        )
        return self


class TestParameterValidation:
    """Validation tests for estimator parameters."""

    @pytest.mark.parametrize("max_history", [0, -1, 1.5])
    def test_invalid_max_history_raises(self, max_history):
        returns = np.ones((5, 4)) * 0.01
        exposures = np.ones_like(returns)
        panel, X = make_panel(returns, extra_fields={"beta": exposures})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            max_history=max_history,
        )

        with pytest.raises(ValueError, match="max_history must be a positive integer"):
            model.fit(X, characteristics=panel)

    @pytest.mark.parametrize(
        ("param", "value", "match"),
        [
            ("benchmark_mcap_power", -1, "benchmark_mcap_power"),
            ("benchmark_mcap_power", True, "benchmark_mcap_power"),
            ("regression_mcap_power", np.nan, "regression_mcap_power"),
            ("regression_mcap_power", "0", "regression_mcap_power"),
            (
                "inv_idio_variance_weight_shrinkage",
                1.5,
                "finite number between 0 and 1",
            ),
            (
                "inv_idio_variance_weight_shrinkage",
                True,
                "finite number between 0 and 1",
            ),
            ("inv_idio_variance_max_weight_ratio", 0, "positive number"),
            ("spanned_alpha_shrinkage", -0.1, "finite number between 0 and 1"),
            ("orthogonal_alpha_confidence", 1.1, "finite number between 0 and 1"),
            ("idio_corr_threshold", 1.1, "finite number between 0 and 1"),
        ],
    )
    def test_invalid_numeric_parameters_raise(self, param, value, match):
        returns = np.ones((5, 4)) * 0.01
        exposures = np.ones_like(returns)
        panel, X = make_panel(returns, extra_fields={"beta": exposures})
        kwargs = {param: value}

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            **kwargs,
        )

        with pytest.raises(ValueError, match=match):
            model.fit(X, characteristics=panel)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("exposure_lag", True),
            ("max_history", True),
            ("min_regression_assets", True),
        ],
    )
    def test_boolean_integer_parameters_raise(self, param, value):
        returns = np.ones((5, 4)) * 0.01
        exposures = np.ones_like(returns)
        panel, X = make_panel(returns, extra_fields={"beta": exposures})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            **{param: value},
        )

        with pytest.raises(ValueError, match="positive integer"):
            model.fit(X, characteristics=panel)

    def test_missing_derived_factor_source_raises_value_error(self):
        returns = np.ones((5, 4)) * 0.01
        panel, X = make_panel(returns)

        model = CharacteristicsFactorModel(
            factors=[
                (
                    "derived",
                    DerivedFactor(
                        source="missing",
                        func=lambda x: x,
                        family="style",
                    ),
                )
            ],
        )

        with pytest.raises(ValueError, match="must be a defined factor"):
            model.fit(X, characteristics=panel)

    def test_factor_name_family_collision_requires_singleton(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        market = rng.normal(size=(n_obs, n_assets))
        beta = rng.normal(size=(n_obs, n_assets))
        panel, X = make_panel(
            returns,
            extra_fields={
                "market": market,
                "beta": beta,
            },
        )

        model = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("market", family="style")),
                ("beta", passthrough_factor("beta", family="market")),
            ],
        )

        with pytest.raises(DuplicateGroupsError, match="family contains"):
            model.fit(X, characteristics=panel)

    def test_self_named_singleton_factor_family_allowed(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        market = rng.normal(size=(n_obs, n_assets))
        beta = rng.normal(size=(n_obs, n_assets))
        panel, X = make_panel(
            returns,
            extra_fields={
                "market": market,
                "beta": beta,
            },
        )

        model = CharacteristicsFactorModel(
            factors=[
                ("market", passthrough_factor("market", family="market")),
                ("beta", passthrough_factor("beta", family="style")),
            ],
            factor_prior_estimator=EmpiricalPrior(
                covariance_estimator=EWCovariance(half_life=2, min_observations=1)
            ),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=n_assets,
        )

        model.fit(X, characteristics=panel)

        factor_model = model.factor_model_
        assert factor_model.factor_names[0] == "market"
        assert factor_model.factor_families[0] == "market"


class TestMaskValidation:
    """Validation tests for mask semantics."""

    def test_trailing_unused_exposure_nan_does_not_fail_regression_coverage(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        exposures = rng.normal(size=(n_obs, n_assets))
        exposures[-1, :2] = np.nan
        panel, X = make_panel(returns, extra_fields={"beta": exposures})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            factor_prior_estimator=EmpiricalPrior(),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=n_assets,
        )

        model.fit(X, characteristics=panel)
        assert model.factor_model_.regression_weights.shape == (n_obs - 1, n_assets)

    def test_factor_covariance_warmup_raises_clear_error(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        exposures = rng.normal(size=(n_obs, n_assets))
        panel, X = make_panel(returns, extra_fields={"beta": exposures})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            factor_prior_estimator=EmpiricalPrior(
                covariance_estimator=EWCovariance(
                    half_life=20,
                    min_observations=50,
                )
            ),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=n_assets,
        )

        with pytest.raises(ValueError, match="finite factor covariance"):
            model.fit(X, characteristics=panel)

    def test_market_cap_required_for_active_assets(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        exposures = rng.normal(size=(n_obs, n_assets))
        market_cap = np.ones((n_obs, n_assets))
        market_cap[:, 0] = np.nan
        panel, X = make_panel(
            returns,
            extra_fields={"beta": exposures},
            market_cap=market_cap,
        )

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            factor_prior_estimator=EmpiricalPrior(),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            exposure_lag=1,
            benchmark_mcap_power=1,
            regression_mcap_power=1,
            min_regression_assets=n_assets - 1,
        )

        with pytest.raises(ValueError, match='Field "market_cap" contains NaN/inf'):
            model.fit(X, characteristics=panel)

    @pytest.mark.parametrize(
        "reserved_field",
        [
            _BENCHMARK_WEIGHTS,
            _EXPOSURES,
            _IDIO_RETURNS,
            _IDIO_VARIANCES,
            _REGRESSION_WEIGHTS,
        ],
    )
    def test_reserved_field_collision_raises(self, reserved_field):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        exposures = rng.normal(size=(n_obs, n_assets))
        panel, X = make_panel(
            returns,
            extra_fields={
                "beta": exposures,
                reserved_field: np.ones_like(returns),
            },
        )

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=n_assets,
        )

        with pytest.raises(ValueError, match=rf"Reserved fields.*{reserved_field}"):
            model.fit(X, characteristics=panel)

    def test_max_history_truncates_sample_weight(self):
        rng = np.random.default_rng(42)
        n_obs, n_assets = 12, 5
        returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
        exposures = rng.normal(size=(n_obs, n_assets))
        panel, X = make_panel(returns, extra_fields={"beta": exposures})

        model = CharacteristicsFactorModel(
            factors=[("beta", passthrough_factor("beta", family="market"))],
            factor_prior_estimator=WeightedEmpiricalPrior(),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=n_assets,
            max_history=5,
        )

        model.fit(X, characteristics=panel)

        assert model.return_distribution_.returns.shape[0] == 5
        np.testing.assert_array_equal(
            model.return_distribution_.sample_weight,
            np.arange(n_obs - 5, n_obs, dtype=float),
        )


class TestResolveNameUnit:
    """Unit tests for the `_resolve_factor_name` helper."""

    FACTOR_TO_IDX: ClassVar[dict[str, int]] = {
        "mkt": 0,
        "size": 1,
        "momentum": 2,
        "ind_A": 3,
        "ind_B": 4,
    }
    FAMILY_TO_IDX: ClassVar[dict[str, list[int]]] = {
        "market": [0],
        "style": [1, 2],
        "industry": [3, 4],
    }

    def test_resolve_single_factor(self):
        result = _resolve_factor_name("size", self.FACTOR_TO_IDX, self.FAMILY_TO_IDX)
        assert result == {1}

    def test_resolve_family(self):
        result = _resolve_factor_name("style", self.FACTOR_TO_IDX, self.FAMILY_TO_IDX)
        assert result == {1, 2}

    def test_resolve_family_industry(self):
        result = _resolve_factor_name(
            "industry", self.FACTOR_TO_IDX, self.FAMILY_TO_IDX
        )
        assert result == {3, 4}

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match=r"'unknown'.*neither a factor name"):
            _resolve_factor_name("unknown", self.FACTOR_TO_IDX, self.FAMILY_TO_IDX)

    def test_factor_name_preferred_over_family(self):
        """If a name is both a factor and a family, factor takes priority."""
        fti = {**self.FACTOR_TO_IDX, "industry": 5}
        result = _resolve_factor_name("industry", fti, self.FAMILY_TO_IDX)
        assert result == {5}


class TestNeutralizeExposureValidation:
    """Unit tests for `_neutralize_exposures` validation logic.

    Uses a minimal synthetic DGP with 5 factors across 3 families
    (market, style, industry) to exercise resolution and overlap checks.
    """

    N_OBS = 10
    N_ASSETS = 8
    FACTOR_NAMES = np.array(["mkt", "size", "momentum", "ind_A", "ind_B"])
    FACTOR_FAMILIES = np.array(["market", "style", "style", "industry", "industry"])

    @staticmethod
    def _make_exposures(rng, n_obs, n_assets, n_factors):
        return rng.standard_normal((n_obs, n_assets, n_factors))

    @staticmethod
    def _make_weights(n_obs, n_assets):
        w = np.ones((n_obs, n_assets))
        return w / w.sum(axis=1, keepdims=True)

    @staticmethod
    def _regressor():
        return CSLinearRegression(fit_intercept=False)

    def test_family_key_neutralizes_all_members(self):
        """Key ``'style'`` must neutralize both size and momentum."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)
        exp_before = exp[:, :, [1, 2]].copy()

        _neutralize_exposures(
            cs_regressor=self._regressor(),
            neutralize_against={"style": ["industry"]},
            exposures=exp,
            benchmark_weights=wt,
            factor_names=self.FACTOR_NAMES,
            factor_families=self.FACTOR_FAMILIES,
        )

        for i, name in [(1, "size"), (2, "momentum")]:
            changed = not np.allclose(exp[:, :, i], exp_before[:, :, i - 1])
            assert changed, f"{name} exposure should have been modified"

    def test_single_factor_key_still_works(self):
        """Key ``'momentum'`` must neutralize only momentum."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)
        size_before = exp[:, :, 1].copy()

        _neutralize_exposures(
            cs_regressor=self._regressor(),
            neutralize_against={"momentum": ["industry"]},
            exposures=exp,
            benchmark_weights=wt,
            factor_names=self.FACTOR_NAMES,
            factor_families=self.FACTOR_FAMILIES,
        )

        np.testing.assert_array_equal(
            exp[:, :, 1], size_before, err_msg="size must not change"
        )

    def test_positive_weight_missing_exposures_are_excluded(self):
        """Positive benchmark weights with missing exposures are excluded."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        exp[2, 0, 1] = np.nan
        exp[3, 1, 3] = np.nan

        _neutralize_exposures(
            cs_regressor=self._regressor(),
            neutralize_against={"size": ["industry"]},
            exposures=exp,
            benchmark_weights=wt,
            factor_names=self.FACTOR_NAMES,
            factor_families=self.FACTOR_FAMILIES,
        )

        assert np.isnan(exp[2, 0, 1])
        assert np.isnan(exp[3, 1, 1])
        valid_size = np.ones((self.N_OBS, self.N_ASSETS), dtype=bool)
        valid_size[2, 0] = False
        valid_size[3, 1] = False
        assert np.all(np.isfinite(exp[:, :, 1][valid_size]))

    def test_overlap_self_neutralization_raises(self):
        """``{'momentum': ['momentum']}`` is direct self-neutralization."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="overlap"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"momentum": ["momentum"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_overlap_family_self_neutralization_raises(self):
        """``{'style': ['style']}`` is family-level self-neutralization."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="overlap"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"style": ["style"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_overlap_family_vs_member_raises(self):
        """``{'style': ['momentum']}`` overlaps because momentum is in style."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="overlap"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"style": ["momentum"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_overlap_mixed_family_and_factor_raises(self):
        """``{'style': ['style', 'industry']}`` overlaps on style members."""
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="overlap"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"style": ["style", "industry"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_unknown_key_raises(self):
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="'nonexistent'"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"nonexistent": ["industry"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_unknown_target_raises(self):
        rng = np.random.default_rng(42)
        exp = self._make_exposures(rng, self.N_OBS, self.N_ASSETS, 5)
        wt = self._make_weights(self.N_OBS, self.N_ASSETS)

        with pytest.raises(ValueError, match="'nonexistent'"):
            _neutralize_exposures(
                cs_regressor=self._regressor(),
                neutralize_against={"style": ["nonexistent"]},
                exposures=exp,
                benchmark_weights=wt,
                factor_names=self.FACTOR_NAMES,
                factor_families=self.FACTOR_FAMILIES,
            )

    def test_family_key_orthogonality(self):
        """After ``{'style': ['industry']}``, the correlation between each
        style and industry exposure must be substantially reduced."""
        rng = np.random.default_rng(42)
        n_obs, n_assets = 100, 50
        exp = self._make_exposures(rng, n_obs, n_assets, 5)
        wt = np.ones((n_obs, n_assets)) / n_assets

        max_corr_before = 0.0
        for si in [1, 2]:
            for ii in [3, 4]:
                c = np.abs(np.corrcoef(exp[0, :, si], exp[0, :, ii])[0, 1])
                max_corr_before = max(max_corr_before, c)

        _neutralize_exposures(
            cs_regressor=self._regressor(),
            neutralize_against={"style": ["industry"]},
            exposures=exp,
            benchmark_weights=wt,
            factor_names=self.FACTOR_NAMES,
            factor_families=self.FACTOR_FAMILIES,
        )

        for t in range(n_obs):
            for si in [1, 2]:
                for ii in [3, 4]:
                    corr = np.abs(np.corrcoef(exp[t, :, si], exp[t, :, ii])[0, 1])
                    assert corr < 0.15, (
                        f"t={t}, style={si}, industry={ii}: corr={corr:.3f}"
                    )
