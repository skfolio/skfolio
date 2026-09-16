"""Tests for reconstructed asset return scenarios."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import clone, config_context

from skfolio import RiskMeasure
from skfolio.moments.variance import EWVariance
from skfolio.optimization import HierarchicalRiskParity
from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior, EntropyPooling
from skfolio.prior._characteristics_factor_model import (
    _assemble_asset_return_scenarios,
    _compute_standardized_idio_returns,
)

from .conftest import make_panel, passthrough_factor


@pytest.mark.parametrize(
    "sample_weight, expected_weight",
    [
        pytest.param([0.4, 0.1, 0.2, 0.3], [1 / 6, 1 / 3, 1 / 2], id="truncated"),
        pytest.param([0.4, 0.0, 0.2, 0.4], [0.0, 1 / 3, 2 / 3], id="zero-weight"),
        pytest.param([0, 0, 0, 1], [0.0, 0.0, 1.0], id="integer-weights"),
        pytest.param(
            [1 - 3e-12, 1e-12, 1e-12, 1e-12],
            [1 / 3, 1 / 3, 1 / 3],
            id="small-retained-mass",
        ),
    ],
)
def test_asset_return_scenarios_rescale_idiosyncratic_shocks_and_align_weights(
    sample_weight, expected_weight
):
    n_assets = 2
    loading_matrix = np.eye(n_assets)
    factor_return_scenarios = np.array(
        [
            [0.10, 0.20],
            [0.30, 0.40],
            [0.50, 0.60],
        ]
    )
    idio_returns = np.array(
        [
            [999.0, 999.0],
            [2.0, 6.0],
            [4.0, 8.0],
            [-6.0, 10.0],
        ]
    )
    idio_variances = np.array(
        [
            [1.0, 1.0],
            [4.0, 9.0],
            [16.0, 16.0],
            [9.0, 25.0],
        ]
    )
    active_mask = np.ones_like(idio_returns, dtype=bool)
    current_idio_variances = np.array([100.0, 400.0])
    sample_weight = np.array(sample_weight)
    original_sample_weight = sample_weight.copy()

    idio_shocks = _compute_standardized_idio_returns(
        idio_returns=idio_returns,
        idio_variances=idio_variances,
        active_mask=active_mask,
    )
    scenarios, aligned_sample_weight = _assemble_asset_return_scenarios(
        factor_return_scenarios=factor_return_scenarios,
        loading_matrix=loading_matrix,
        standardized_idio_returns=idio_shocks,
        latest_active_mask=np.ones(n_assets, dtype=bool),
        latest_idio_variances=current_idio_variances,
        sample_weight=sample_weight,
    )

    expected_idio_scenarios = np.array(
        [
            [10.0, 40.0],
            [10.0, 40.0],
            [-20.0, 40.0],
        ]
    )
    np.testing.assert_allclose(
        scenarios, factor_return_scenarios + expected_idio_scenarios
    )
    np.testing.assert_allclose(aligned_sample_weight, expected_weight)
    np.testing.assert_array_equal(sample_weight, original_sample_weight)


def test_asset_return_scenarios_zero_retained_sample_weight():
    with pytest.raises(
        ValueError, match="Retained scenarios must have positive total sample weight"
    ):
        _assemble_asset_return_scenarios(
            factor_return_scenarios=np.zeros((3, 1)),
            loading_matrix=np.ones((1, 1)),
            standardized_idio_returns=np.zeros((2, 1)),
            latest_active_mask=np.array([True]),
            latest_idio_variances=np.ones(1),
            sample_weight=np.array([1.0, 0.0, 0.0]),
        )


def test_truncated_sample_weight_with_hierarchical_risk_parity():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, size=(12, 5))
    panel, X = make_panel(
        returns,
        extra_fields={
            "beta": rng.normal(size=returns.shape),
            "beta2": rng.normal(size=returns.shape),
        },
    )
    with config_context(enable_metadata_routing=True):
        prior = CharacteristicsFactorModel(
            factors=[
                ("beta", passthrough_factor("beta", family="market")),
                ("beta2", passthrough_factor("beta2", family="style")),
            ],
            factor_prior_estimator=EntropyPooling(),
            idio_variance_estimator=EWVariance(half_life=2, min_observations=1),
            exposure_lag=1,
            benchmark_mcap_power=0,
            regression_mcap_power=0,
            min_regression_assets=5,
            max_history=5,
        ).set_fit_request(characteristics=True)
        model = HierarchicalRiskParity(
            prior_estimator=prior, risk_measure=RiskMeasure.VARIANCE
        )
        model.fit(X, characteristics=panel)

        # With no views, entropy pooling should match the unweighted empirical prior.
        reference = clone(model).set_params(
            prior_estimator__factor_prior_estimator=EmpiricalPrior()
        )
        reference.fit(X, characteristics=panel)

    distribution = model.prior_estimator_.return_distribution_
    assert distribution.returns.shape == (5, 5)
    np.testing.assert_allclose(distribution.sample_weight, np.full(5, 0.2))
    np.testing.assert_allclose(model.weights_, reference.weights_)


def test_asset_return_scenarios_impute_missing_active_residuals_from_active_assets():
    n_assets = 3
    loading_matrix = np.eye(n_assets)
    factor_return_scenarios = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 2.0, 6.0],
        ]
    )
    idio_returns = np.array(
        [
            [2.0, np.nan, 100.0],
            [np.nan, np.nan, 5.0],
        ]
    )
    idio_variances = np.array(
        [
            [4.0, np.nan, 1.0],
            [np.nan, np.nan, 1.0],
        ]
    )
    active_mask = np.array(
        [
            [True, True, False],
            [True, True, False],
        ]
    )
    current_idio_variances = np.array([100.0, 400.0, 900.0])
    current_active_mask = active_mask[-1]

    idio_shocks = _compute_standardized_idio_returns(
        idio_returns=idio_returns,
        idio_variances=idio_variances,
        active_mask=active_mask,
    )
    scenarios, sample_weight = _assemble_asset_return_scenarios(
        factor_return_scenarios=factor_return_scenarios,
        loading_matrix=loading_matrix,
        standardized_idio_returns=idio_shocks,
        latest_active_mask=current_active_mask,
        latest_idio_variances=current_idio_variances,
        sample_weight=None,
    )

    expected = np.array(
        [
            [10.0, 20.0, np.nan],
            [1.0, 2.0, np.nan],
        ]
    )
    np.testing.assert_allclose(scenarios, expected, equal_nan=True)
    assert sample_weight is None
