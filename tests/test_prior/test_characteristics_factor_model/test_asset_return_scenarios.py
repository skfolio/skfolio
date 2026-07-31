"""Tests for reconstructed asset return scenarios."""

from __future__ import annotations

import numpy as np

from skfolio.prior._characteristics_factor_model import (
    _assemble_asset_return_scenarios,
    _compute_standardized_idio_returns,
)


def test_asset_return_scenarios_rescale_idiosyncratic_shocks_and_align_weights():
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
    sample_weight = np.array([0.0, 1.0, 2.0, 3.0])

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
    np.testing.assert_array_equal(aligned_sample_weight, sample_weight[-3:])


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
