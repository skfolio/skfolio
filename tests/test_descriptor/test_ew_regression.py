"""Independent numerical, lifecycle and uncertainty checks for matched EW fits."""

import inspect
import warnings
from decimal import Decimal, localcontext
from itertools import pairwise

import numpy as np
import pytest

from skfolio.containers import AssetPanel, Field2D, FieldCategorical, InactivePolicy
from skfolio.descriptor import (
    EWDownsideBeta,
    EWMacroSensitivity,
    EWMarketBeta,
    EWResidualDownsideVolatility,
    EWResidualVolatility,
    Passthrough,
)
from skfolio.descriptor._utils import _EWRegression
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.utils.tools import half_life_to_decay_factor
from tests.test_descriptor._ew_reference import (
    downside_path,
    regression_path,
    residual_path,
    weighted_regression,
)


def _panel(market, returns, active=None, groups=False):
    """Keep the benchmark fixed by using only the first asset to construct it."""
    returns = np.column_stack((market, returns))
    if active is None:
        active = np.ones(returns.shape, dtype=bool)
    returns[~active] = np.nan
    estimation = np.zeros(returns.shape, dtype=bool)
    estimation[:, 0] = True
    fields = {"returns": returns, "market_cap": np.ones(returns.shape)}
    fields["market_cap"][~active] = np.nan
    if groups:
        fields["industry"] = FieldCategorical(
            values=np.where(active, 0, -1), levels=np.array(["all"])
        )
    return AssetPanel(
        fields=fields,
        observations=np.arange(len(market)),
        asset_names=np.array([f"asset_{i}" for i in range(returns.shape[1])]),
        active_mask=active,
        estimation_mask=estimation,
    )


def _fit_regression(returns, half_life, **kwargs):
    """Feed a complete one-asset history to the numerical helper."""
    regression = _EWRegression(
        n_assets=1, decay=half_life_to_decay_factor(half_life), **kwargs
    )
    for row in returns:
        regression.update(row[:, None], np.array([True]))
    return regression


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("aggregation", [1, 5])
@pytest.mark.parametrize("half_life", [0.5, 7.3, 60])
def test_lifecycle_matches_batch_and_chunks(reference, aggregation, half_life):
    rng = np.random.default_rng(501)
    market = rng.normal(0, 0.01, 103)
    market[40:] *= 3
    factor = rng.normal(0, 0.01, len(market))
    returns = 0.003 + 1.7 * market[:, None] + rng.normal(0, 0.004, (len(market), 3))
    if reference:
        returns += 0.6 * factor[:, None]
        factor[[23, 24, 46, 47, 71]] = np.nan
    active = np.ones((len(market), 4), dtype=bool)
    active[:14, 1] = False
    active[46:48, 2] = False  # Reset even when the reference is missing.
    active[70:73, 3] = False
    returns[[19, 20, 31, 51, 72], 0] = np.nan
    returns[60:65, 1] = np.nan  # An entirely empty aggregation window.
    panel = _panel(market, returns, active)
    cls = EWMacroSensitivity if reference else EWMarketBeta
    params = dict(half_life=half_life, aggregation_period=aggregation, min_periods=4)
    fit_params = dict(reference_returns=factor) if reference else {}
    expected = regression_path(
        panel, half_life, 4, aggregation, factor if reference else None
    )
    estimator = cls(**params)
    result = estimator.fit_transform(panel, **fit_params)
    np.testing.assert_allclose(result, expected, atol=2e-10, rtol=2e-10, equal_nan=True)
    online = cls(**params)
    chunks = []
    split_indices = [0, 1, 17, 24, 47, 48, 63, 74, 101, 103]
    for start, stop in pairwise(split_indices):
        kwargs = dict(reference_returns=factor[start:stop]) if reference else {}
        chunks.append(online.partial_fit_transform(panel[start:stop], **kwargs))
    np.testing.assert_array_equal(np.vstack(chunks), result)
    np.testing.assert_array_equal(estimator.fit_transform(panel, **fit_params), result)


@pytest.mark.parametrize("volatility_change", [0.5, 1, 2])
@pytest.mark.parametrize("cls", [EWMarketBeta, EWDownsideBeta, EWMacroSensitivity])
def test_issue_501_exact_relation(volatility_change, cls):
    """Late listings recover exact slopes across market volatility changes."""
    # Late-listing regression coverage originated in PR #507 by Vinayak19112003.
    rng = np.random.default_rng(501)
    market = rng.normal(0, 0.01, 200)
    market[80:] *= volatility_change
    reference = rng.normal(0, 0.01, len(market))
    returns = 2 * market
    kwargs = {}
    if cls is EWMacroSensitivity:
        returns += 1.5 * reference + 0.003
        kwargs["reference_returns"] = reference
    elif cls is EWMarketBeta:
        returns += 0.003
    active = np.ones((len(market), 2), dtype=bool)
    active[:80, 1] = False
    result = cls(half_life=60).fit_transform(_panel(market, returns, active), **kwargs)
    assert np.isnan(result[:139, 1]).all()
    np.testing.assert_allclose(result[139:, 1], 1.5 if kwargs else 2, atol=2e-14)


@pytest.mark.parametrize("n", [3, 4, 10, 60, 120, 600])
@pytest.mark.parametrize("half_life", [0.5, 4, 60, 150])
def test_squared_weight_uncertainty_matches_matrix_formula(n, half_life):
    rng = np.random.default_rng(5)
    market = rng.normal(size=n)
    market[-20:] *= 3
    returns = np.column_stack((market, 0.7 + 1.5 * market + rng.normal(size=n)))
    regression = _fit_regression(returns, half_life, estimate_beta_variance=True)
    beta = regression.beta()
    expected_beta, expected_variance = weighted_regression(returns, half_life)
    np.testing.assert_allclose(beta, expected_beta, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(
        regression.beta_variance(beta), expected_variance, rtol=5e-11
    )


def test_uncertainty_is_calibrated_with_changing_market_volatility():
    rng = np.random.default_rng(17)
    n_paths, n = 10000, 60
    market = rng.normal(size=n)
    market[-20:] *= 3
    returns = 0.3 + 1.5 * market[:, None] + rng.normal(size=(n, n_paths))
    regression = _EWRegression(
        n_assets=n_paths,
        decay=half_life_to_decay_factor(60),
        estimate_beta_variance=True,
    )
    valid = np.ones(n_paths, dtype=bool)
    for t in range(n):
        regression.update(np.vstack((np.full(n_paths, market[t]), returns[t])), valid)
    beta = regression.beta()
    np.testing.assert_allclose(
        np.mean(regression.beta_variance(beta)), np.var(beta), rtol=0.04
    )


@pytest.mark.parametrize("relative_spread", [1e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-6])
def test_rounding_jitter_against_decimal(relative_spread):
    rng = np.random.default_rng(8)
    market = 1 + relative_spread * rng.normal(size=60)
    target = rng.normal(size=60)
    regression = _fit_regression(np.column_stack((market, target)), 60)
    with localcontext() as context:
        context.prec = 70
        decay = Decimal.from_float(regression.decay)
        weights = [decay**k for k in range(59, -1, -1)]
        mass = sum(weights)
        x = [Decimal.from_float(v) for v in market]
        y = [Decimal.from_float(v) for v in target]
        mx = sum(w * v for w, v in zip(weights, x, strict=True)) / mass
        my = sum(w * v for w, v in zip(weights, y, strict=True)) / mass
        scatter = sum(w * (v - mx) ** 2 for w, v in zip(weights, x, strict=True))
        covariance = sum(
            w * (v - mx) * (r - my) for w, v, r in zip(weights, x, y, strict=True)
        )
        expected = float(covariance / scatter)
    if relative_spread <= 1e-15:
        assert np.isnan(regression.beta()[0])
    else:
        np.testing.assert_allclose(regression.beta(), expected, rtol=2e-13)


@pytest.mark.parametrize("cls", [EWMarketBeta, EWMacroSensitivity])
@pytest.mark.parametrize("level", [-1.0, 1.0, 1e-120, 1e155])
@pytest.mark.parametrize(
    "relative_spread, has_variation",
    [(16 * np.finfo(float).eps, False), (48 * np.finfo(float).eps, True)],
)
def test_relative_variation_boundary(cls, level, relative_spread, has_variation):
    """The variation threshold follows predictor scale without squaring its level."""
    predictor = level + np.tile([-1, 1], 30) * abs(level) * relative_spread
    target = 2 * (predictor - level)
    if cls is EWMacroSensitivity:
        panel = _panel(np.zeros(60), target)
        kwargs = {"reference_returns": predictor}
    else:
        panel = _panel(predictor, target)
        kwargs = {}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = cls(min_periods=2).fit_transform(panel, **kwargs)
    assert np.isnan(result[0, 1])
    if has_variation:
        np.testing.assert_allclose(result[1:, 1], 2, atol=1e-13)
    else:
        assert np.isnan(result[:, 1]).all()


@pytest.mark.parametrize(
    "market_scale, reference_scale", [(1e-150, 1e150), (1e150, 1e-150)]
)
def test_macro_sensitivity_with_different_return_scales(market_scale, reference_scale):
    """An exact exposure remains estimable when return scales differ substantially."""
    market_noise, reference_noise = np.random.default_rng(507).normal(size=(2, 80))
    market = market_scale * (1 + 1e-10 * market_noise)
    reference = reference_scale * (0.5 * market_noise + reference_noise)
    panel = _panel(market, 1.5 * reference)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = EWMacroSensitivity(half_life=6, min_periods=3).fit_transform(
            panel, reference_returns=reference
        )

    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:, 1], 1.5, rtol=2e-12)


def test_long_level_change_preserves_small_recent_variation():
    rng = np.random.default_rng(2)
    market = np.r_[rng.normal(size=100), 1 + rng.normal(size=3000) * 1e-10]
    target = 0.1 + 1.5 * market
    regression = _fit_regression(np.column_stack((market, target)), 5)
    np.testing.assert_allclose(
        regression.beta(),
        weighted_regression(np.column_stack((market[-200:], target[-200:])), 5)[0],
        rtol=2e-6,
    )


@pytest.mark.parametrize("market_type", ["zero", "constant", "unresolved", "varying"])
def test_macro_constant_market_reduction(market_type):
    rng = np.random.default_rng(6)
    reference = rng.normal(0, 0.01, 80)
    market = np.zeros(80)
    if market_type == "constant":
        market[:] = 0.01
    elif market_type == "unresolved":
        market = 1 + rng.normal(size=80) * np.spacing(1.0)
    elif market_type == "varying":
        market = reference + rng.normal(0, 0.01, 80)
    panel = _panel(market, 0.005 + 2 * market + 1.5 * reference)
    result = EWMacroSensitivity(min_periods=4).fit_transform(
        panel, reference_returns=reference
    )
    if market_type == "unresolved":
        assert np.isnan(result[:, 1]).all()
    else:
        np.testing.assert_allclose(result[3:, 1], 1.5, rtol=1e-12, atol=1e-14)


def test_macro_decayed_variation_does_not_trigger_constant_market_reduction():
    """Underflow of past market variation keeps the two-predictor fit unidentified."""
    rng = np.random.default_rng(6)
    market = np.r_[rng.normal(0, 0.01, 20), np.full(600, 0.01)]
    reference = rng.normal(0, 0.01, len(market))
    result = EWMacroSensitivity(half_life=0.5, min_periods=4).fit_transform(
        _panel(market, 0.005 + 2 * market + 1.5 * reference),
        reference_returns=reference,
    )
    np.testing.assert_allclose(result[3:20, 1], 1.5, rtol=1e-12)
    assert np.isnan(result[-1, 1])


def test_beta_variance_requires_market_only_regression():
    with pytest.raises(ValueError, match="include_macro_returns=False"):
        _EWRegression(
            n_assets=1,
            decay=0.9,
            include_macro_returns=True,
            estimate_beta_variance=True,
        )


@pytest.mark.parametrize("gap", [0, 0.5e-8, 2e-8, 1e-4])
def test_macro_collinearity_threshold(gap):
    rng = np.random.default_rng(8)
    market = rng.normal(size=100)
    independent = rng.normal(size=100)
    weights = 2.0 ** (-np.arange(99, -1, -1) / 60)
    market -= np.average(market, weights=weights)
    independent -= np.average(independent, weights=weights)
    independent -= (
        market
        * np.average(market * independent, weights=weights)
        / np.average(market**2, weights=weights)
    )
    independent *= np.sqrt(
        np.average(market**2, weights=weights)
        / np.average(independent**2, weights=weights)
    )
    reference = np.sqrt(1 - gap) * market + np.sqrt(gap) * independent
    returns = np.column_stack((market, reference, 2 * market + 1.5 * reference))
    regression = _fit_regression(returns, 60, include_macro_returns=True)
    if gap < 1e-8:
        assert np.isnan(regression.beta()).all()
    else:
        np.testing.assert_allclose(regression.beta(), 1.5, rtol=2e-7)


def test_rejected_beta_replaces_cache_and_recovers():
    market = np.r_[[-1, 1], np.ones(100), [-1, 1]]
    result = EWMarketBeta(half_life=0.5, min_periods=2).fit_transform(
        _panel(market, 2 * market)
    )
    assert result[1, 1] == pytest.approx(2)
    assert np.isnan(result[100, 1])
    assert result[-1, 1] == pytest.approx(2)


@pytest.mark.parametrize(
    "cls",
    [
        EWMarketBeta,
        EWMacroSensitivity,
        EWDownsideBeta,
        EWResidualVolatility,
        EWResidualDownsideVolatility,
    ],
)
def test_eps_warning_once_per_fit_and_half_life_resolution(cls):
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, 30)
    panel = _panel(market, market + rng.normal(0, 0.01, 30))
    reference = rng.normal(0, 0.01, 30)
    kwargs = {"reference_returns": reference} if cls is EWMacroSensitivity else {}
    expected = cls(min_periods=3).fit_transform(panel, **kwargs)
    estimator = cls(min_periods=3, eps=0.1)
    with pytest.warns(FutureWarning, match="eps is deprecated") as records:
        result = estimator.fit_transform(panel, **kwargs)
    assert len(records) == 1
    assert records[0].filename == __file__
    np.testing.assert_array_equal(result, expected)
    with warnings.catch_warnings(record=True) as records:
        estimator.partial_fit_transform(panel, **kwargs)
    assert len(records) == 0
    with pytest.warns(FutureWarning, match="eps is deprecated"):
        estimator.fit_transform(panel, **kwargs)
    with pytest.raises(ValueError, match="rounds to zero"):
        cls(half_life=1e30).fit_transform(panel, **kwargs)
    for method in ("fit_transform", "partial_fit_transform"):
        estimator = cls(min_periods=3, eps=0.1)
        with pytest.warns(FutureWarning, match="eps is deprecated") as records:
            line = inspect.currentframe().f_lineno + 1
            getattr(estimator, method)(panel, **kwargs)
        assert records[0].filename == __file__
        assert records[0].lineno == line


@pytest.mark.parametrize("cls", [EWResidualVolatility, EWResidualDownsideVolatility])
@pytest.mark.parametrize("beta_half_life", [0.5, 1, 3, 4, 5, 20, 60])
def test_residual_support_and_lifecycle(cls, beta_half_life):
    rng = np.random.default_rng(9)
    market = rng.normal(0, 0.01, 100)
    returns = 0.001 + market[:, None] + rng.normal(0, 0.002, (100, 2))
    active = np.ones((100, 3), dtype=bool)
    active[45:47, 1] = False
    returns[35:38, 1] = np.nan
    panel = _panel(market, returns, active)
    estimator = cls(half_life=4, beta_half_life=beta_half_life, min_periods=3)
    result = estimator.fit_transform(panel)
    mar = 0 if cls is EWResidualDownsideVolatility else None
    expected = residual_path(panel, 4, beta_half_life, 3, mar)
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=1e-12, equal_nan=True)
    assert np.isnan(result[:11]).all()
    assert np.isfinite(result[11]).all()
    assert np.isnan(result[47:58, 1]).all()
    assert np.isfinite(result[58, 1])
    online = cls(half_life=4, beta_half_life=beta_half_life, min_periods=3)
    np.testing.assert_array_equal(
        np.vstack(
            [online.partial_fit_transform(panel[i : i + 7]) for i in range(0, 100, 7)]
        ),
        result,
    )


def test_residual_keeps_intercept_and_excludes_two_point_startup():
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, 100)
    market[:2] = [0.01, 0.01 + 1e-14]
    target = 0.003 + 2 * market
    target[:2] += [-0.001, 0.001]
    panel = _panel(market, target)
    result = EWResidualVolatility(half_life=40, beta_half_life=60).fit_transform(panel)
    expected = residual_path(panel, 40, 60, 60)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
    assert np.isnan(result[:68]).all()
    assert 0.0029 < result[68, 1] < 0.0031
    exact = EWResidualVolatility(half_life=40, beta_half_life=60).fit_transform(
        _panel(market, 0.003 + 2 * market)
    )
    np.testing.assert_allclose(exact[68:, 1], 0.003, atol=1e-15)


def test_downside_lifecycle_and_constant_negative_market():
    rng = np.random.default_rng(2)
    market = rng.normal(0, 0.01, 100)
    active = np.ones((100, 3), dtype=bool)
    active[50:54, 1] = False
    returns = market[:, None] + rng.normal(0, 0.01, (100, 2))
    returns[10:16, 1] = np.nan
    panel = _panel(market, returns, active)
    result = EWDownsideBeta(
        half_life=4.3, min_periods=3, min_acceptable_return=0.002
    ).fit_transform(panel)
    expected = downside_path(panel, 4.3, 3, 0.002)
    np.testing.assert_allclose(result, expected, rtol=1e-13, equal_nan=True)
    online = EWDownsideBeta(half_life=4.3, min_periods=3, min_acceptable_return=0.002)
    np.testing.assert_array_equal(
        np.vstack(
            [online.partial_fit_transform(panel[i : i + 7]) for i in range(0, 100, 7)]
        ),
        result,
    )
    constant = EWDownsideBeta(min_periods=1).fit_transform(
        _panel(np.full(5, -0.01), np.full(5, -0.02))
    )
    np.testing.assert_allclose(constant[:, 1], 2)


def test_shrinkage_unknown_uncertainty_and_group_formula():
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, 60)
    returns = market[:, None] * np.arange(1, 7) + rng.normal(0, 0.02, (60, 6))
    panel = _panel(market, returns, groups=True)
    estimator = EWMarketBeta(
        half_life=20,
        min_periods=2,
        shrinkage_group="industry",
        shrinkage_bounds=(0.1, 0.9),
    )
    result = estimator.fit_transform(panel)
    assert np.isnan(result[:2]).all()  # A two-point slope has no uncertainty estimate.
    slopes, variances = [], []
    for i in range(panel.n_assets):
        beta, variance = weighted_regression(
            np.column_stack((market, panel["returns"][:, i])), 20
        )
        slopes.append(beta[0])
        variances.append(variance[0])
    slopes, variances = np.array(slopes), np.array(variances)
    mean = slopes.mean()
    prior = max(slopes.var() - variances.mean(), 0)
    raw_weight = np.clip(
        prior / (prior + variances) if prior > 0 else np.zeros_like(slopes), 0.1, 0.9
    )
    np.testing.assert_allclose(
        result[-1], raw_weight * slopes + (1 - raw_weight) * mean, rtol=1e-12
    )
    online = EWMarketBeta(
        half_life=20,
        min_periods=2,
        shrinkage_group="industry",
        shrinkage_bounds=(0.1, 0.9),
    )
    np.testing.assert_array_equal(
        np.vstack(
            [online.partial_fit_transform(panel[i : i + 7]) for i in range(0, 60, 7)]
        ),
        result,
    )


def test_shrinkage_exemptions_and_zero_prior():
    market = np.array([-0.01, 0.01, -0.005, 0.02])
    panel = _panel(market, np.column_stack((2 * market, 2 * market)), groups=True)
    panel["industry"][1, 1] = -1
    panel["market_cap"][1, 2] = 0
    result = EWMarketBeta(min_periods=2, shrinkage_group="industry").fit_transform(
        panel
    )
    assert np.isnan(result[1, 0])
    np.testing.assert_allclose(result[1, 1:], 2)
    identical = _panel(market, market, groups=True)
    result = EWMarketBeta(min_periods=3, shrinkage_group="industry").fit_transform(
        identical
    )
    np.testing.assert_allclose(result[2:], 1)


def test_arithmetic_failure_is_explicit():
    regression = _EWRegression(n_assets=1, decay=half_life_to_decay_factor(5))
    regression.update(np.array([[1e200], [1e200]]), np.array([True]))
    with pytest.raises(ValueError, match="Non-finite EW regression moments"):
        regression.update(np.array([[-1e200], [-1e200]]), np.array([True]))


def test_gaussian_startup_tail_40000_paths():
    """Ten-pair beta warmup bounds the residual startup tail in this Gaussian case."""
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, (40000, 240)).T
    returns = (0.0005 + market.T + rng.normal(0, 0.015, (40000, 240))).T
    regression = _EWRegression(n_assets=40000, decay=half_life_to_decay_factor(60))
    residuals = []
    for t in range(69):
        regression.update(
            np.vstack((market[t], returns[t])), np.ones(40000, dtype=bool)
        )
        if t >= 9:
            residuals.append(returns[t] - regression.beta(10) * market[t])
    assert len(residuals) == 60
    weights = 2.0 ** (-np.arange(59, -1, -1) / 40)
    rms = np.sqrt(np.average(np.square(residuals), weights=weights, axis=0))
    assert np.quantile(rms / np.hypot(0.0005, 0.015), 0.999) < 1.5


@pytest.mark.parametrize("beta_half_life", [1, 3, 4, 5, 20, 60])
@pytest.mark.parametrize("vol_half_life", [10, 40, 120])
def test_residual_heavy_tail_sensitivity(beta_half_life, vol_half_life):
    """Varied decays and t5 noise retain actual large contributions without clipping."""
    rng = np.random.default_rng(771)
    market = rng.normal(0.003, 0.01, 170)
    market[20:] = 0.003 + 3 * (market[20:] - 0.003)
    returns = (
        0.0005 + market[:, None] + rng.standard_t(5, (170, 3)) * 0.015 * np.sqrt(3 / 5)
    )
    panel = _panel(market, returns)
    result = EWResidualVolatility(
        half_life=vol_half_life, beta_half_life=beta_half_life
    ).fit_transform(panel)
    expected = residual_path(
        panel, vol_half_life, beta_half_life, max(vol_half_life, beta_half_life)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
    assert np.isfinite(result[-1]).all()


@pytest.mark.parametrize("coverage", [0, 0.5, 0.75])
def test_unidentifiable_descriptor_factor_coverage(coverage):
    panel = _panel(np.full(10, 0.01), np.full(10, 0.02))
    panel["benchmark_weights"] = np.full((10, 2), 0.5)
    factor = FixedWeightedFactor(
        descriptors=[
            ("beta", EWMarketBeta(min_periods=2)),
            ("cap", Passthrough("market_cap")),
        ],
        min_coverage=coverage,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )
    result = factor.fit_transform(panel)
    if coverage <= 0.5:
        np.testing.assert_array_equal(result, 1)
    else:
        assert np.isnan(result).all()
    factor.set_params(descriptors=[("beta", EWMarketBeta(min_periods=2))])
    assert np.isnan(factor.fit_transform(panel)).all()


@pytest.mark.parametrize(
    "cls", [EWMarketBeta, EWMacroSensitivity, EWDownsideBeta, EWResidualVolatility]
)
def test_inactive_finite_rows_are_ignored(cls):
    rng = np.random.default_rng(43)
    market = rng.normal(0, 0.01, 80)
    returns = market + rng.normal(0, 0.01, 80)
    active = np.ones((80, 2), dtype=bool)
    active[30:33, 1] = False
    expected_panel = _panel(market, returns, active)
    finite_panel = _panel(market, returns)
    finite_panel["returns"] = Field2D(
        finite_panel["returns"], inactive_policy=InactivePolicy.IGNORE
    )
    finite_panel["market_cap"] = Field2D(
        finite_panel["market_cap"], inactive_policy=InactivePolicy.IGNORE
    )
    finite_panel.active_mask = active
    kwargs = {}
    if cls is EWMacroSensitivity:
        reference = rng.normal(0, 0.01, 80)
        reference[30:33] = np.nan
        kwargs["reference_returns"] = reference
    expected = cls(min_periods=3).fit_transform(expected_panel, **kwargs)
    result = cls(min_periods=3).fit_transform(finite_panel, **kwargs)
    np.testing.assert_array_equal(result, expected)


def test_minimum_rank_constant_target_and_zero_decay():
    market = np.array([-0.01, 0.01, 0.03, -0.02])
    panel = _panel(market, np.full(4, 0.005))
    result = EWMarketBeta(min_periods=1).fit_transform(panel)
    assert np.isnan(result[0]).all()
    np.testing.assert_array_equal(result[1:, 1], 0)
    regression = _fit_regression(
        np.column_stack((market, np.full(4, 0.005))), 60, estimate_beta_variance=True
    )
    np.testing.assert_array_equal(regression.beta_variance(regression.beta()), 0)
    result = EWMarketBeta(half_life=1e-10, min_periods=1).fit_transform(panel)
    assert np.isnan(result).all()
    reference = np.array([0.01, -0.01, 0.02, 0.03])
    result = EWMacroSensitivity(min_periods=1).fit_transform(
        panel, reference_returns=reference
    )
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:, 1], 0, atol=1e-14)


def test_uncertainty_missing_rows_and_reset_match_batch():
    rng = np.random.default_rng(22)
    regression = _EWRegression(
        n_assets=2, decay=half_life_to_decay_factor(6), estimate_beta_variance=True
    )
    histories = [[], []]
    for t in range(50):
        returns = rng.normal(size=(2, 2))
        active = np.array([t != 25, True])
        valid = active & np.array([t % 5 != 0, t % 7 != 0])
        regression.reset(~active)
        regression.update(returns, valid)
        beta = regression.beta()
        variance = regression.beta_variance(beta)
        for i in range(2):
            if not active[i]:
                histories[i].clear()
            if valid[i]:
                histories[i].append(returns[:, i])
            if len(histories[i]) >= 3:
                expected_beta, expected_variance = weighted_regression(histories[i], 6)
                np.testing.assert_allclose(beta[i], expected_beta[0], atol=1e-14)
                np.testing.assert_allclose(
                    variance[i], expected_variance[0], rtol=1e-12
                )


def test_residual_numerical_rejection_freezes_history_then_recovers():
    """Residual history survives rejection and recovery under the relative threshold."""
    rng = np.random.default_rng(5)
    market = np.r_[rng.normal(0, 0.01, 30), np.full(400, 0.01), rng.normal(0, 0.01, 30)]
    target = 0.003 + 2 * market + rng.normal(0, 0.01, len(market))
    panel = _panel(market, target)
    estimator = EWResidualVolatility(half_life=40, beta_half_life=4, min_periods=3)
    initial = estimator.partial_fit_transform(panel[:430])
    assert np.isnan(initial[-1]).all()
    recovered = estimator.partial_fit_transform(panel[430:])
    result = np.vstack((initial, recovered))[:, 1]

    expected = np.full(len(market), np.nan)
    residuals = []
    for t in range(9, len(market)):
        weights = 2.0 ** (-np.arange(t, -1, -1) / 4)
        weights /= weights.sum()
        # Anchor at the current observation to preserve variation during flat stretches.
        shifted = market[: t + 1] - market[t]
        mean_shift = weights @ shifted
        centered = shifted - mean_shift
        variance = weights @ centered**2
        mean = market[t] + mean_shift
        if np.sqrt(variance) > 32 * np.finfo(float).eps * abs(mean):
            shifted_target = target[: t + 1] - target[t]
            centered_target = shifted_target - weights @ shifted_target
            beta = (weights @ (centered * centered_target)) / variance
            residuals.append(target[t] - beta * market[t])
            if len(residuals) >= 3:
                residual_weights = 2.0 ** (-np.arange(len(residuals) - 1, -1, -1) / 40)
                expected[t] = np.sqrt(
                    np.average(np.square(residuals), weights=residual_weights)
                )

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
    assert np.isfinite(recovered[:, 1]).all()
    np.testing.assert_array_equal(
        result,
        EWResidualVolatility(
            half_life=40, beta_half_life=4, min_periods=3
        ).fit_transform(panel)[:, 1],
    )
