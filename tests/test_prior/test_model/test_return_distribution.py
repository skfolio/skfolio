"""Tests for the return-distribution value object."""

import numpy as np
import pytest

from skfolio.prior import FactorModel, ReturnDistribution


def _make_return_distribution(**overrides) -> ReturnDistribution:
    params = {
        "mu": np.array([0.01, 0.02]),
        "covariance": np.array([[0.04, 0.01], [0.01, 0.09]]),
        "returns": np.array([[0.01, 0.02], [-0.01, 0.03], [0.02, -0.01]]),
        "sample_weight": np.array([0.2, 0.3, 0.5]),
    }
    params.update(overrides)
    return ReturnDistribution(**params)


def test_return_distribution_reports_investable_assets():
    """Investability follows finite means and covariance diagonal entries."""
    distribution = _make_return_distribution()

    assert distribution.n_assets == 2
    assert distribution.n_investable_assets == 2
    assert distribution.investable_mask is None

    distribution = _make_return_distribution(mu=np.array([0.01, np.nan]))

    np.testing.assert_array_equal(distribution.investable_mask, [True, False])
    assert distribution.n_investable_assets == 1

    covariance = np.array([[0.04, 0.01], [0.01, np.nan]])
    distribution = _make_return_distribution(covariance=covariance)

    np.testing.assert_array_equal(distribution.investable_mask, [True, False])
    assert distribution.n_investable_assets == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"mu": np.ones((1, 2))}, "mu.*1D"),
        ({"covariance": np.eye(3)}, "covariance.*shape"),
        ({"returns": np.ones((3, 3))}, "returns.*shape"),
        ({"sample_weight": np.ones(2)}, "sample_weight.*shape"),
    ],
)
def test_return_distribution_rejects_incompatible_shapes(overrides, match):
    """All arrays must align with the asset and observation dimensions."""
    with pytest.raises(ValueError, match=match):
        _make_return_distribution(**overrides)


def test_return_distribution_requires_an_investable_asset():
    """A distribution with no finite expected return cannot be optimized."""
    distribution = _make_return_distribution(mu=np.full(2, np.nan))

    with pytest.raises(ValueError, match="All assets are non-investable"):
        _ = distribution.investable_mask


@pytest.mark.parametrize(
    "weights",
    [
        [-0.1, 0.3, 0.8],
        [np.nan, 0.5, 0.5],
        [np.inf, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0.2j, 0.3, 0.5],
    ],
)
def test_return_distribution_rejects_invalid_probabilities(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        _make_return_distribution(sample_weight=np.array(weights))


def test_return_distribution_unnormalized_weight_error():
    with pytest.raises(ValueError) as error:
        _make_return_distribution(sample_weight=np.array([1, 2, 3]))
    assert str(error.value) == (
        "sample_weight must sum to one, got 6.0. Normalize the weights "
        "before constructing the portfolio or return distribution."
    )


def test_return_distribution_normalizes_within_tolerance_without_mutation():
    weights = np.array([0.2, 0.3, 0.5]) * (1 + 1e-6)
    original = weights.copy()
    weights.setflags(write=False)
    distribution = _make_return_distribution(sample_weight=weights)
    np.testing.assert_allclose(distribution.sample_weight.sum(), 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(weights, original)
    np.testing.assert_array_equal(distribution.mu, [0.01, 0.02])


def test_return_distribution_allows_empty_probability_vector():
    distribution = _make_return_distribution(
        returns=np.empty((0, 2)), sample_weight=np.array([])
    )
    assert distribution.sample_weight.shape == (0,)


def _make_factor_model(n_assets: int) -> FactorModel:
    return FactorModel(
        observations=np.arange(3),
        asset_names=np.array([f"asset_{i}" for i in range(n_assets)]),
        factor_names=np.array(["f0"]),
        factor_families=None,
        loading_matrix=np.ones((n_assets, 1)),
        exposures=None,
        factor_covariance=np.array([[0.01]]),
        factor_mu=np.array([0.001]),
        factor_returns=None,
        idio_covariance=np.full(n_assets, 0.02),
        idio_mu=None,
        idio_returns=None,
        idio_variances=None,
    )


def test_return_distribution_rejects_factor_model_on_other_universe():
    """The factor model loading matrix must cover the same assets as `mu`."""
    with pytest.raises(ValueError, match="same asset universe"):
        _make_return_distribution(factor_model=_make_factor_model(n_assets=3))


def test_investable_subset_slices_factor_model_along_assets():
    """Non-investable assets are dropped from the nested factor model as well."""
    distribution = _make_return_distribution(
        mu=np.array([0.01, np.nan]), factor_model=_make_factor_model(n_assets=2)
    )

    subset = distribution.investable_subset()

    assert subset.n_assets == 1
    np.testing.assert_array_equal(subset.factor_model.asset_names, ["asset_0"])
    assert subset.factor_model.loading_matrix.shape == (1, 1)
    np.testing.assert_array_equal(subset.factor_model.idio_covariance, [0.02])
