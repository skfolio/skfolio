from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.containers import AssetPanel, FieldCategorical
from skfolio.factor_model.descriptor import (
    EWMacroSensitivity,
    EWMarketBeta,
    Passthrough,
)
from skfolio.factor_model.factor_exposure import FixedWeightedFactor, GlobalFactor
from skfolio.optimization import MeanRisk
from skfolio.prior import CharacteristicsFactorModel
from tests.test_factor_model._characteristics_data import (
    generate_synthetic_characteristics,
)


def _make_characteristics_factor_model_inputs():
    characteristics = generate_synthetic_characteristics(
        n_assets=80,
        years=1,
        seed=42,
        delist_prob=0.0,
        list_late_prob=0.0,
        mid_nan_frac=0.0,
    )
    characteristics.ffill("market_cap").bfill("market_cap")
    X = characteristics.to_dataframe(fields="returns")

    rng = np.random.default_rng(123)
    reference_returns = rng.normal(0.0, 0.005, size=characteristics.n_observations)
    return X, characteristics, reference_returns


def _make_characteristics_factor_model():
    macro_descriptor = EWMacroSensitivity(
        half_life=3,
        min_periods=3,
        aggregation_period=1,
    )
    factors = [
        ("global", GlobalFactor(family="market")),
        (
            "fx",
            FixedWeightedFactor(
                descriptors=[("macro", macro_descriptor)],
                family="style",
                outlier_transformer="passthrough",
                scoring_transformer="passthrough",
            ),
        ),
    ]
    return CharacteristicsFactorModel(factors=factors)


def test_passthrough_descriptor(simple_panel):
    """Test that passthrough descriptors return raw panel fields."""
    factor = FixedWeightedFactor(
        descriptors=[("market_cap", Passthrough("market_cap"))],
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )

    result = factor.fit_transform(simple_panel)

    np.testing.assert_allclose(result, simple_panel["market_cap"])
    assert isinstance(factor.descriptors_[0], Passthrough)


def test_string_descriptor_specs_raise(simple_panel):
    """Test that descriptor specs must be descriptor estimators."""
    factor = FixedWeightedFactor(
        descriptors=[("market_cap", "market_cap")],
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )

    with pytest.raises(TypeError, match="Expected descriptor to be a BaseDescriptor"):
        factor.fit_transform(simple_panel)


def test_multi_descriptor_passthrough_scoring(simple_panel):
    """Test multi-descriptor weighted averages when scoring is skipped."""
    weights = np.array([0.25, 0.75])
    factor = FixedWeightedFactor(
        descriptors=[
            ("market_cap", Passthrough("market_cap")),
            ("book_equity", Passthrough("book_equity")),
        ],
        weights=weights,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )

    result = factor.fit_transform(simple_panel)
    expected = (
        weights[0] * simple_panel["market_cap"]
        + weights[1] * simple_panel["book_equity"]
    )

    np.testing.assert_allclose(result, expected)


def test_min_coverage_masks_low_coverage_cells(simple_panel):
    """Test that min_coverage requires enough valid descriptor weight."""
    simple_panel["book_equity"][0, 0] = np.nan
    factor = FixedWeightedFactor(
        descriptors=[
            ("market_cap", Passthrough("market_cap")),
            ("book_equity", Passthrough("book_equity")),
        ],
        weights=np.array([0.4, 0.6]),
        min_coverage=0.5,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )

    result = factor.fit_transform(simple_panel)

    assert np.isnan(result[0, 0])
    np.testing.assert_allclose(
        result[0, 1],
        0.4 * simple_panel["market_cap"][0, 1]
        + 0.6 * simple_panel["book_equity"][0, 1],
    )


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([[1.0, 0.0]], "1D array"),
        ([1.0], "same length"),
        ([0.2, 0.3], "sum to 1"),
        ([-1.0, 2.0], "non-negative"),
    ],
)
def test_invalid_weights_raise(simple_panel, weights, match):
    """Test descriptor weight validation."""
    factor = FixedWeightedFactor(
        descriptors=[
            ("market_cap", Passthrough("market_cap")),
            ("book_equity", Passthrough("book_equity")),
        ],
        weights=weights,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )

    with pytest.raises(ValueError, match=match):
        factor.fit_transform(simple_panel)


def _assert_characteristics_factor_model_outputs_equal(left, right):
    np.testing.assert_array_equal(
        left.factor_model_.factor_names,
        right.factor_model_.factor_names,
    )
    np.testing.assert_allclose(
        left.factor_model_.exposures,
        right.factor_model_.exposures,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        left.return_distribution_.mu,
        right.return_distribution_.mu,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        left.return_distribution_.covariance,
        right.return_distribution_.covariance,
        equal_nan=True,
    )


def _assert_prior_outputs_equal(left, right):
    _assert_characteristics_factor_model_outputs_equal(
        left.prior_estimator_,
        right.prior_estimator_,
    )
    np.testing.assert_allclose(left.weights_, right.weights_, equal_nan=True)


def _assert_fx_exposures_are_finite(model):
    assert "fx" in model.factor_model_.factor_names
    fx_idx = list(model.factor_model_.factor_names).index("fx")
    fx_exposures = model.factor_model_.exposures[..., fx_idx]
    assert np.isfinite(fx_exposures).any()


def test_grouped_warmup_descriptor_preserves_nan_rows():
    rng = np.random.default_rng(42)
    n_observations = 20
    n_assets = 20

    market_returns = rng.normal(0.0, 0.01, size=n_observations)
    betas = np.linspace(0.5, 1.5, n_assets)
    returns = market_returns[:, None] * betas[None, :] + rng.normal(
        0.0, 0.005, size=(n_observations, n_assets)
    )
    market_cap = np.full((n_observations, n_assets), 1.0)
    benchmark_weights = np.full((n_observations, n_assets), 1.0 / n_assets)
    industry = np.broadcast_to(
        np.repeat([0, 1], n_assets // 2)[None, :],
        (n_observations, n_assets),
    )

    panel = AssetPanel(
        fields={
            "returns": returns,
            "market_cap": market_cap,
            "benchmark_weights": benchmark_weights,
            "industry": FieldCategorical(
                industry.astype(np.int32, copy=False),
                levels=np.array(["group_0", "group_1"]),
            ),
        },
        observations=np.arange(n_observations),
        asset_names=np.array([f"asset_{i}" for i in range(n_assets)]),
        active_mask=np.ones((n_observations, n_assets), dtype=bool),
        estimation_mask=np.ones((n_observations, n_assets), dtype=bool),
    )

    factor = FixedWeightedFactor(
        descriptors=[
            ("market_beta", EWMarketBeta(half_life=3, shrinkage_group=None)),
        ],
        transform_by_group="industry",
    )

    exposures = factor.fit_transform(panel)

    finite_rows = np.isfinite(exposures).any(axis=1)
    assert np.any(finite_rows)
    first_finite_row = np.flatnonzero(finite_rows)[0]
    assert first_finite_row > 0
    assert np.all(np.isnan(exposures[:first_finite_row]))
    assert np.all(np.isfinite(exposures[first_finite_row:]))


def test_characteristics_factor_model_fit_routes_descriptor_metadata():
    X, characteristics, reference_returns = _make_characteristics_factor_model_inputs()
    with config_context(enable_metadata_routing=True):
        model = _make_characteristics_factor_model()
        model.fit(
            X, characteristics=characteristics, reference_returns=reference_returns
        )

    _assert_fx_exposures_are_finite(model)


def test_characteristics_factor_model_partial_fit_routes_descriptor_metadata():
    X, characteristics, reference_returns = _make_characteristics_factor_model_inputs()
    with config_context(enable_metadata_routing=True):
        model = _make_characteristics_factor_model()
        model.partial_fit(
            X,
            characteristics=characteristics,
            reference_returns=reference_returns,
        )

    _assert_fx_exposures_are_finite(model)


def test_mean_risk_routes_characteristics_and_descriptor_metadata_by_default():
    X, characteristics, reference_returns = _make_characteristics_factor_model_inputs()
    with config_context(enable_metadata_routing=True):
        batch = MeanRisk(prior_estimator=_make_characteristics_factor_model())
        batch.fit(
            X,
            characteristics=characteristics,
            reference_returns=reference_returns,
        )

        online = MeanRisk(prior_estimator=_make_characteristics_factor_model())
        online.partial_fit(
            X,
            characteristics=characteristics,
            reference_returns=reference_returns,
        )

    _assert_prior_outputs_equal(batch, online)


def test_characteristics_factor_model_partial_fit_rejects_initial_warmup_only_batch():
    X, characteristics, reference_returns = _make_characteristics_factor_model_inputs()
    with config_context(enable_metadata_routing=True):
        model = _make_characteristics_factor_model()
        with pytest.raises(ValueError, match="Not enough observations"):
            model.partial_fit(
                X.iloc[:1],
                characteristics=characteristics[:1],
                reference_returns=reference_returns[:1],
            )

    assert not hasattr(model, "factor_model_")
    assert not hasattr(model, "return_distribution_")


def test_characteristics_factor_model_accepts_one_date_update_after_initial_fit():
    X, characteristics, reference_returns = _make_characteristics_factor_model_inputs()
    initial = slice(0, -1)
    update = slice(-1, None)

    with config_context(enable_metadata_routing=True):
        batch = _make_characteristics_factor_model()
        batch.fit(
            X,
            characteristics=characteristics,
            reference_returns=reference_returns,
        )

        online = _make_characteristics_factor_model()
        online.partial_fit(
            X.iloc[initial],
            characteristics=characteristics[initial],
            reference_returns=reference_returns[initial],
        )
        online.partial_fit(
            X.iloc[update],
            characteristics=characteristics[update],
            reference_returns=reference_returns[update],
        )

    _assert_characteristics_factor_model_outputs_equal(batch, online)
