"""Test Measure module."""

from __future__ import annotations

import numpy as np
import pytest

import skfolio.measures as skm
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

WEIGHTED_RETURN_MEASURES = [
    skm.mean,
    skm.mean_absolute_deviation,
    skm.first_lower_partial_moment,
    skm.variance,
    skm.semi_variance,
    skm.standard_deviation,
    skm.semi_deviation,
    skm.third_central_moment,
    skm.skew,
    skm.fourth_central_moment,
    skm.kurtosis,
    skm.value_at_risk,
    skm.cvar,
    skm.entropic_risk_measure,
]


@pytest.mark.parametrize(
    "measure,kwargs,expected",
    [
        (skm.mean, {}, [0.16, 0.4, np.nan]),
        (skm.variance, {"biased": True}, [0.0024, 0.0, np.nan]),
        (skm.variance, {}, [0.005, np.nan, np.nan]),
        (skm.semi_variance, {"biased": True}, [0.00144, 0.0, np.nan]),
        (skm.semi_variance, {}, [0.003, np.nan, np.nan]),
    ],
)
def test_weighted_moments_normalize_each_column(measure, kwargs, expected):
    returns = np.array([[0.1, np.nan, np.nan], [0.2, 0.4, np.nan]])
    # The first column keeps weights [0.4, 0.6]; the second keeps only weight 1.
    # The first column's unbiased correction is 1 - 0.4**2 - 0.6**2 = 0.48.
    np.testing.assert_allclose(
        measure(returns, sample_weight=np.array([0.4, 0.6]), **kwargs),
        expected,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("measure", WEIGHTED_RETURN_MEASURES)
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("two_dimensional", [False, True])
def test_weighted_measures_ignore_missing_returns(measure, uniform, two_dimensional):
    returns = np.array(
        [
            [np.nan, 0.1, np.nan],
            [0.1, np.nan, np.nan],
            [-0.5, -0.5, np.nan],
            [np.nan, 0.3, np.nan],
            [np.nan, 0.2, np.nan],
            [-0.3, -0.3, np.nan],
            [0.8, 0.8, np.nan],
        ]
    )
    weights = np.ones(7) if uniform else np.arange(1.0, 8.0)
    weights /= weights.sum()
    if not two_dimensional:
        returns = returns[:, 0]
    expected = []
    columns = returns.T if two_dimensional else [returns]
    for column in columns:
        valid = ~np.isnan(column)
        if not valid.any():
            expected.append(np.nan)
        else:
            remaining = weights[valid] / weights[valid].sum()
            expected.append(measure(column[valid], sample_weight=remaining))
    expected = np.array(expected) if two_dimensional else expected[0]
    original = returns.copy()
    original_weights = weights.copy()
    returns.setflags(write=False)
    weights.setflags(write=False)

    actual = measure(returns, sample_weight=weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    if uniform:
        np.testing.assert_allclose(actual, measure(returns), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(returns, original)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("measure", WEIGHTED_RETURN_MEASURES)
@pytest.mark.parametrize(
    "returns,weights",
    [
        ([], []),
        ([np.nan, np.nan], [0.2, 0.8]),
        ([0.1, np.nan], [0.0, 1.0]),
    ],
)
def test_weighted_measures_without_usable_mass(measure, returns, weights):
    assert np.isnan(measure(np.array(returns), sample_weight=np.array(weights)))


@pytest.mark.parametrize("measure", [skm.variance, skm.semi_variance])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("tiny", [None, 1e-200, 1e-320])
@pytest.mark.parametrize("two_dimensional", [False, True])
def test_weighted_second_moments_after_missing_returns(
    measure, biased, tiny, two_dimensional
):
    returns = np.array([[0.01, np.nan], [0.03, 0.02], [np.nan, 0.04]])
    weights = np.array([tiny, tiny, 1.0]) if tiny else np.array([0.2, 0.3, 0.5])
    kwargs = {"biased": biased}
    if measure is skm.semi_variance:
        kwargs["min_acceptable_return"] = np.array([0.04, 0.01])
    if not two_dimensional:
        returns = returns[:, 0]
        if measure is skm.semi_variance:
            kwargs["min_acceptable_return"] = 0.04
    actual = measure(returns, sample_weight=weights, **kwargs)
    columns = returns.T if two_dimensional else [returns]
    expected = []
    for j, column in enumerate(columns):
        valid = ~np.isnan(column)
        remaining = weights[valid] / weights[valid].sum()
        args = kwargs.copy()
        if measure is skm.semi_variance and two_dimensional:
            args["min_acceptable_return"] = kwargs["min_acceptable_return"][j]
        expected.append(measure(column[valid], sample_weight=remaining, **args))
    np.testing.assert_allclose(actual, expected if two_dimensional else expected[0])


@pytest.mark.parametrize("tiny", [1e-200, 1e-320])
def test_weighted_mean_with_tiny_surviving_mass(tiny):
    returns = np.array([0.01, 0.03, np.nan])
    weights = np.array([tiny, tiny, 1.0])
    np.testing.assert_allclose(skm.mean(returns, sample_weight=weights), 0.02)
    np.testing.assert_allclose(
        skm.mean(returns[:, None], sample_weight=weights), [0.02]
    )


@pytest.mark.parametrize("measure", [skm.variance, skm.semi_variance])
def test_unbiased_second_moment_requires_two_positive_weights(measure):
    kwargs = {"min_acceptable_return": 0.1} if measure is skm.semi_variance else {}
    assert np.isnan(
        measure(np.array([0.01, np.nan]), sample_weight=np.array([0.2, 0.8]), **kwargs)
    )
    assert np.isnan(
        measure(np.array([0.01, 0.02]), sample_weight=np.array([1.0, 0.0]), **kwargs)
    )
    # The scalar and column paths must agree even if rounding leaves a tiny
    # positive correction for the column with only one usable observation.
    result = measure(
        np.array([[0.01, 0.02], [np.nan, 0.04]]),
        sample_weight=np.array([0.43, 0.57]),
        **kwargs,
    )
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


@pytest.mark.parametrize("measure", WEIGHTED_RETURN_MEASURES)
def test_weighted_measures_empty_matrix(measure):
    np.testing.assert_array_equal(
        measure(np.empty((2, 0)), sample_weight=np.array([0.2, 0.8])), []
    )


@pytest.mark.parametrize("measure", [skm.value_at_risk, skm.cvar])
def test_weighted_tail_missing_mass_and_fractional_boundary(measure):
    # The worst return carries exactly the tail mass 1 - beta, so the VaR is the
    # next loss and the CVaR is the worst loss.
    expected = -0.1 if measure is skm.value_at_risk else 0.1
    np.testing.assert_allclose(
        measure(
            np.array([-0.1, 0.1, np.nan]),
            beta=0.5,
            sample_weight=np.array([0.25, 0.25, 0.5]),
        ),
        expected,
    )
    # Only half of the second observation belongs to the lower 50% tail.
    expected = 0.1 if measure is skm.value_at_risk else 0.22
    np.testing.assert_allclose(
        measure(
            np.array([-0.3, -0.1, 0.2, np.nan]),
            beta=0.5,
            sample_weight=np.array([0.15, 0.20, 0.15, 0.50]),
        ),
        expected,
    )


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_weighted_var_endpoints_ignore_zero_mass(beta):
    returns = np.array([-0.5, 0.1, 0.2, 0.9, np.nan])
    weights = np.array([0.0, 0.2, 0.3, 0.0, 0.5])
    expected = -0.1 if beta == 1 else -0.2
    assert skm.value_at_risk(returns, beta=beta, sample_weight=weights) == expected


def test_weighted_tail_zero_beta_rounding():
    rng = np.random.default_rng(0)
    returns = rng.normal(size=11)
    weights = rng.random(11)
    weights /= weights.sum()
    assert skm.value_at_risk(returns, beta=0, sample_weight=weights) == -returns.max()
    np.testing.assert_allclose(
        skm.cvar(returns, beta=0, sample_weight=weights), -weights @ returns
    )


@pytest.fixture(scope="module")
def returns_1d():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices[["AAPL"]], log_returns=False)
    returns_1d = X.to_numpy().reshape(-1)
    return returns_1d


@pytest.fixture(scope="module")
def returns_2d():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices[["AAPL", "AMD"]], log_returns=False)
    returns_2d = X.to_numpy()
    return returns_2d


@pytest.fixture(scope="module")
def returns_1d_nan():
    returns_1d_nan = np.array([np.nan, 0.1, -0.5, np.nan, np.nan, -0.3, 0.8])
    return returns_1d_nan


@pytest.fixture(scope="module")
def returns_2d_nan():
    returns_2d_nan = np.array(
        [
            [0.15, 0.1, -0.5, np.nan, np.nan, -0.3, 0.8],
            [0.15, 0.1, -0.5, 0.3, 0.2, -0.3, 0.8],
        ]
    ).T
    return returns_2d_nan


@pytest.fixture(scope="module")
def returns_all_nan(returns_1d):
    returns_all_nan = np.full(len(returns_1d), np.nan)
    return returns_all_nan


@pytest.fixture(
    scope="module",
    params=["1d", "2d", "1d_nan", "2d_nan", "all_nan"],
)
def returns(
    request, returns_1d, returns_2d, returns_1d_nan, returns_2d_nan, returns_all_nan
):
    match request.param:
        case "1d":
            return returns_1d
        case "2d":
            return returns_2d
        case "1d_nan":
            return returns_1d_nan
        case "2d_nan":
            return returns_2d_nan
        case "all_nan":
            return returns_all_nan
        case _:
            raise ValueError(f"request.param {request.param} not found")


@pytest.fixture(scope="module", params=[True, False])
def sample_weight(request, returns_1d):
    if not request.param:
        return
    rng = np.random.default_rng(0)
    sample_weight = rng.random(len(returns_1d))
    sample_weight /= sample_weight.sum()
    return sample_weight


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, 0.0011233),
        ("1d", True, 0.0010021),
        ("2d", False, [0.0011234, 0.0010841]),
        ("2d", True, [0.0010021, 0.0010231]),
        ("1d_nan", False, 0.025),
        ("2d_nan", False, [0.05, 0.1071429]),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_mean(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.mean(returns, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_mean_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.mean(returns, sample_weight=q),
        skm.mean(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,min_acceptable_return,sample_weight,expected",
    [
        ("1d", None, False, 0.0186077),
        ("1d", 0.0, False, 0.0185881),
        ("1d", None, True, 0.0185890),
        ("1d", 0.0, True, 0.0185672),
        ("2d", None, False, [0.0186077, 0.0268726]),
        ("2d", 0.0, False, [0.0185882, 0.0268155]),
        ("2d", None, True, [0.0185890, 0.0267258]),
        ("2d", 0.0, True, [0.0185672, 0.0266662]),
        ("1d_nan", None, False, 0.425),
        ("all_nan", None, False, np.nan),
        ("all_nan", None, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_mean_absolute_deviation(
    returns, min_acceptable_return, sample_weight, expected
):
    np.testing.assert_almost_equal(
        skm.mean_absolute_deviation(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=sample_weight,
        ),
        expected,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("min_acceptable_return", [None, 0.0])
def test_mean_absolute_deviation_sample_weight(returns, min_acceptable_return):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.mean_absolute_deviation(
            returns, min_acceptable_return=min_acceptable_return, sample_weight=q
        ),
        skm.mean_absolute_deviation(
            returns, min_acceptable_return=min_acceptable_return
        ),
        10,
    )


@pytest.mark.parametrize(
    "returns,min_acceptable_return,sample_weight,expected",
    [
        ("1d", None, False, 0.0093038),
        ("1d", 0.0, False, 0.0087324),
        ("1d", None, True, 0.0092945),
        ("1d", 0.0, True, 0.0087825),
        ("2d", None, False, [0.0093039, 0.0134363]),
        ("2d", 0.0, False, [0.0087324, 0.0128657]),
        ("2d", None, True, [0.0092945, 0.0133629]),
        ("2d", 0.0, True, [0.0087826, 0.0128216]),
        ("1d_nan", None, False, 0.2125),
        ("all_nan", None, False, np.nan),
        ("all_nan", None, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_first_lower_partial_moment(
    returns, min_acceptable_return, sample_weight, expected
):
    np.testing.assert_almost_equal(
        skm.first_lower_partial_moment(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=sample_weight,
        ),
        expected,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("min_acceptable_return", [None, 0.0])
def test_first_lower_partial_moment_sample_weight(returns, min_acceptable_return):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.first_lower_partial_moment(
            returns, min_acceptable_return=min_acceptable_return, sample_weight=q
        ),
        skm.first_lower_partial_moment(
            returns, min_acceptable_return=min_acceptable_return
        ),
        10,
    )


@pytest.mark.parametrize(
    "returns,sample_weight,biased,expected",
    [
        ("1d", False, True, 0.0007478807),
        ("1d", False, False, 0.0007479707),
        ("1d", True, True, 0.0007406331),
        ("1d", True, False, 0.0007407522),
        ("2d", False, True, [0.0007478807, 0.0015151899]),
        ("2d", False, False, [0.0007479707, 0.00151537226]),
        ("2d", True, True, [0.0007406331, 0.0015184352]),
        ("2d", True, False, [0.0007407523, 0.0015186795]),
        ("1d_nan", False, True, 0.246875),
        ("2d_nan", False, True, [0.2, 0.1517346939]),
        ("all_nan", False, True, np.nan),
        ("all_nan", True, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_variance(returns, sample_weight, biased, expected):
    np.testing.assert_almost_equal(
        skm.variance(returns, sample_weight=sample_weight, biased=biased), expected, 10
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("biased", [True, False])
def test_variance_sample_weight(returns, biased):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.variance(returns, sample_weight=q, biased=biased),
        skm.variance(returns, biased=biased),
        10,
    )


@pytest.mark.parametrize(
    "returns,min_acceptable_return,sample_weight,biased,expected",
    [
        ("1d", None, False, False, 0.0003683217),
        ("1d", None, False, True, 0.0003682774),
        ("1d", None, True, False, 0.0003650834),
        ("1d", None, True, True, 0.0003650247),
        ("1d", 0.0, False, False, 0.0003480616),
        ("1d", 0.0, False, True, 0.0003480197),
        ("2d", None, False, False, [0.0003683217, 0.0007148891]),
        ("2d", None, False, True, [0.0003682774, 0.0007148031]),
        ("2d", None, True, False, [0.0003650835, 0.0007130392]),
        ("2d", None, True, True, [0.0003650247, 0.0007129245]),
        ("2d", 0.0, False, False, [0.0003480616, 0.000686374]),
        ("2d", 0.0, False, True, [0.0003480198, 0.0006862914]),
        ("1d_nan", None, False, True, 0.0953125),
        ("1d_nan", None, False, False, 0.1270833333),
        ("2d_nan", None, False, True, [0.085, 0.0763483965]),
        ("2d_nan", None, False, False, [0.10625, 0.0890731293]),
        ("all_nan", None, False, True, np.nan),
        ("all_nan", None, False, False, np.nan),
        ("all_nan", None, True, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_semi_variance(returns, min_acceptable_return, sample_weight, biased, expected):
    np.testing.assert_almost_equal(
        skm.semi_variance(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=sample_weight,
            biased=biased,
        ),
        expected,
        10,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.parametrize("min_acceptable_return", [None, 0.0])
def test_semi_variance_sample_weight(returns, biased, min_acceptable_return):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.semi_variance(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=q,
            biased=biased,
        ),
        skm.semi_variance(
            returns, min_acceptable_return=min_acceptable_return, biased=biased
        ),
        10,
    )


@pytest.mark.parametrize(
    "returns,sample_weight,biased,expected",
    [
        ("1d", False, True, 0.0273474),
        ("1d", False, False, 0.0273490),
        ("1d", True, True, 0.0272145),
        ("1d", True, False, 0.0272167),
        ("2d", False, True, [0.0273474, 0.0389254]),
        ("2d", False, False, [0.0273490, 0.0389277]),
        ("2d", True, True, [0.0272146, 0.0389671]),
        ("2d", True, False, [0.0272168, 0.0389702]),
        ("1d_nan", False, True, 0.4968651728),
        ("all_nan", False, True, np.nan),
        ("all_nan", True, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_standard_deviation(returns, sample_weight, biased, expected):
    np.testing.assert_almost_equal(
        skm.standard_deviation(returns, sample_weight=sample_weight, biased=biased),
        expected,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("biased", [True, False])
def test_standard_deviation_sample_weight(returns, biased):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.standard_deviation(returns, sample_weight=q, biased=biased),
        skm.standard_deviation(returns, biased=biased),
        10,
    )


@pytest.mark.parametrize(
    "returns,min_acceptable_return,sample_weight,biased,expected",
    [
        ("1d", None, False, False, 0.0191917),
        ("1d", None, False, True, 0.0191905),
        ("1d", None, True, False, 0.0191071),
        ("1d", None, True, True, 0.0191056),
        ("1d", 0.0, False, False, 0.0186564),
        ("1d", 0.0, False, True, 0.01865528),
        ("2d", None, False, False, [0.01919177, 0.02673741]),
        ("2d", None, False, True, [0.0191906, 0.0267358]),
        ("2d", None, True, False, [0.0191072, 0.0267028]),
        ("2d", None, True, True, [0.0191056, 0.0267006]),
        ("2d", 0.0, False, False, [0.0186564, 0.02619866]),
        ("2d", 0.0, False, True, [0.0186553, 0.0261972]),
        ("1d_nan", None, False, True, 0.308727225),
        ("1d_nan", None, False, False, 0.3564874939),
        ("2d_nan", None, False, True, [0.2915475947, 0.276312136]),
        ("2d_nan", None, False, False, [0.3259601203, 0.2984512175]),
        ("all_nan", None, False, True, np.nan),
        ("all_nan", None, False, False, np.nan),
        ("all_nan", None, True, True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_semi_deviation(
    returns, min_acceptable_return, sample_weight, biased, expected
):
    np.testing.assert_almost_equal(
        skm.semi_deviation(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=sample_weight,
            biased=biased,
        ),
        expected,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.parametrize("min_acceptable_return", [None, 0.0])
def test_semi_deviation_sample_weight(returns, biased, min_acceptable_return):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.semi_deviation(
            returns,
            min_acceptable_return=min_acceptable_return,
            sample_weight=q,
            biased=biased,
        ),
        skm.semi_deviation(
            returns, min_acceptable_return=min_acceptable_return, biased=biased
        ),
        10,
    )


@pytest.mark.parametrize("measure", [skm.semi_variance, skm.semi_deviation])
def test_semi_measures_nan_correction_matches_dropping_nans(measure):
    # Padding a series with NaNs must not change the result.
    returns = np.array([np.nan, 0.1, -0.5, np.nan, np.nan, -0.3, 0.8])
    np.testing.assert_almost_equal(
        measure(returns), measure(returns[~np.isnan(returns)]), 10
    )

    returns_2d = np.array(
        [
            [0.15, 0.1, -0.5, np.nan, np.nan, -0.3, 0.8],
            [0.15, 0.1, -0.5, 0.3, 0.2, -0.3, 0.8],
        ]
    ).T
    for i in range(returns_2d.shape[1]):
        column = returns_2d[:, i]
        np.testing.assert_almost_equal(
            measure(returns_2d)[i], measure(column[~np.isnan(column)]), 10
        )


@pytest.mark.parametrize("measure", [skm.semi_variance, skm.semi_deviation])
def test_semi_measures_insufficient_observations(measure):
    # Match variance's behavior for fewer than two valid observations.
    assert np.isnan(skm.variance(np.array([0.1])))
    assert np.isnan(measure(np.array([0.1])))
    assert np.isnan(measure(np.array([np.nan, 0.1, np.nan])))


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, -8.0259e-06),
        ("1d", True, -1.04409e-05),
        ("2d", False, [-8.0259e-06, 2.02371e-05]),
        ("2d", True, [-1.04409e-05, 2.18454e-05]),
        ("1d_nan", False, 0.07171875),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_third_central_moment(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.third_central_moment(returns, sample_weight=sample_weight), expected, 10
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_third_central_moment_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.third_central_moment(returns, sample_weight=q),
        skm.third_central_moment(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, -0.3924154),
        ("1d", True, -0.5180085),
        ("2d", False, [-0.3924155, 0.3431228]),
        ("2d", True, [-0.5180086, 0.3692041]),
        ("1d_nan", False, 0.58467838959),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_skew(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.skew(returns, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_skew_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.skew(returns, sample_weight=q),
        skm.skew(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, 1.47128822e-05),
        ("1d", True, 1.40494576e-05),
        ("2d", False, [1.47128822e-05, 2.95098431e-05]),
        ("2d", True, [1.40494576e-05, 2.93127320e-05]),
        ("1d_nan", False, 0.11197695312),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_fourth_central_moment(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.fourth_central_moment(returns, sample_weight=sample_weight), expected, 10
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_fourth_central_moment_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.fourth_central_moment(returns, sample_weight=q),
        skm.fourth_central_moment(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, 26.3046784),
        ("1d", True, 25.6125784),
        ("2d", False, [26.3046784, 12.8538355]),
        ("2d", True, [25.6125784, 12.7134604]),
        ("1d_nan", False, 1.83727607755),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_kurtosis(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.kurtosis(returns, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_kurtosis_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.kurtosis(returns, sample_weight=q),
        skm.kurtosis(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,min_acceptable_return,expected",
    [
        ("1d", None, 1.0675109e-05),
        ("1d", 0.0, 1.0510848e-05),
        ("2d", None, [1.0675109e-05, 1.1070756e-05]),
        ("2d", 0.0, [1.0510848e-05, 1.0777481e-05]),
        ("1d_nan", None, 0.0217814),
        ("all_nan", None, np.nan),
    ],
    indirect=["returns"],
)
def test_fourth_lower_partial_moment(returns, min_acceptable_return, expected):
    (
        np.testing.assert_almost_equal(
            skm.fourth_lower_partial_moment(
                returns, min_acceptable_return=min_acceptable_return
            ),
            expected,
        ),
        10,
    )


@pytest.mark.parametrize(
    "returns,expected",
    [
        ("1d", 0.5184729),
        ("2d", [0.5184729, 0.3793103]),
    ],
    indirect=["returns"],
)
def test_worst_return(returns, expected):
    np.testing.assert_almost_equal(skm.worst_realization(returns), expected)


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, 0.039568345),
        ("1d", True, 0.039682539),
        ("2d", False, [0.039568345, 0.05453306]),
        ("2d", True, [0.039682539, 0.05424528]),
        ("1d_nan", False, 0.5),
        ("2d_nan", False, [0.5, 0.5]),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_value_at_risk(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.value_at_risk(returns, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_value_at_risk_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.value_at_risk(returns, sample_weight=q),
        skm.value_at_risk(returns),
        10,
    )


@pytest.mark.parametrize(
    "n_observations,beta,rank",
    [
        (20, 0.95, 2),
        (100, 0.99, 2),
        (100, 0.95, 6),
        (200, 0.95, 11),
        (1000, 0.99, 11),
        (1000, 0.975, 26),
        (10, 0.9, 2),
        (100, 0.8, 21),
        (252, 0.95, 13),
        (100, 0.995, 1),
        (100, 0.95 + 1e-12, 5),
        (100, 0.95 - 1e-12, 6),
    ],
)
def test_value_at_risk_integer_tail_size(n_observations, beta, rank):
    # When (1 - beta) * n_observations is an integer k, the VaR is the (k + 1)-th
    # largest loss even though that product is inexact in floating point, e.g.
    # (1 - 0.95) * 100 == 5.000000000000004 and (1 - 0.9) * 10 == 0.9999999999999998.
    # Offsetting beta by 1e-12 moves the tail size across the boundary.
    losses = np.random.default_rng(42).permutation(
        np.arange(1, n_observations + 1, dtype=float)
    )
    expected = n_observations - rank + 1
    np.testing.assert_almost_equal(skm.value_at_risk(-losses, beta=beta), expected)
    q = np.ones(n_observations) / n_observations
    np.testing.assert_almost_equal(
        skm.value_at_risk(-losses, beta=beta, sample_weight=q), expected
    )
    np.testing.assert_almost_equal(skm.drawdown_at_risk(-losses, beta=beta), expected)


def test_value_at_risk_integer_tail_size_2d():
    rng = np.random.default_rng(42)
    losses = np.arange(1, 101, dtype=float)
    returns = np.column_stack([-rng.permutation(losses), -rng.permutation(losses)])
    np.testing.assert_almost_equal(skm.value_at_risk(returns, beta=0.95), [95, 95])
    q = np.ones(100) / 100
    np.testing.assert_almost_equal(
        skm.value_at_risk(returns, beta=0.95, sample_weight=q), [95, 95]
    )


def test_value_at_risk_integer_tail_size_nan():
    # NaN returns are excluded: the second column has 20 valid returns, so k = 1.
    rng = np.random.default_rng(42)
    col = np.full(100, np.nan)
    col[:20] = -rng.permutation(np.arange(1, 21, dtype=float))
    returns = np.column_stack([-rng.permutation(np.arange(1, 101, dtype=float)), col])
    np.testing.assert_almost_equal(skm.value_at_risk(col, beta=0.95), 19)
    np.testing.assert_almost_equal(skm.value_at_risk(returns, beta=0.95), [95, 19])
    q = np.ones(100) / 100
    np.testing.assert_almost_equal(
        skm.value_at_risk(returns, beta=0.95, sample_weight=q), [95, 19]
    )


@pytest.mark.parametrize(
    "beta,small_weight,large_weight",
    [(0.9, 0.02, 0.08), (0.95, 0.01, 0.09)],
)
def test_value_at_risk_sample_weight_integer_tail_mass(
    beta, small_weight, large_weight
):
    # The five largest losses carry a total weight of 1 - beta, so the VaR is the
    # sixth largest loss even though the cumulative weights are inexact.
    losses = np.arange(1, 21, dtype=float)
    sample_weight = np.where(losses > 10, small_weight, large_weight)
    np.testing.assert_almost_equal(
        skm.value_at_risk(-losses, beta=beta, sample_weight=sample_weight), 15
    )


@pytest.mark.parametrize(
    "first_weight,expected",
    [
        (1 / 16 - 2**-35, 0.01),
        (1 / 16, 0.01),
        (1 / 16 + 2**-35, 0.20),
    ],
)
def test_value_at_risk_sample_weight_tail_boundary(first_weight, expected):
    # The worst observation has a weight just below, equal to, or just above the
    # tail probability 1 - beta = 1/16. All values are exactly representable.
    returns = np.array([-0.20, -0.01])
    sample_weight = np.array([first_weight, 1 - first_weight])
    np.testing.assert_almost_equal(
        skm.value_at_risk(returns, beta=0.9375, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, 0.059240073),
        ("1d", True, 0.058665975),
        ("2d", False, [0.0592401, 0.0852369]),
        ("2d", True, [0.0586659, 0.0852807]),
        ("1d_nan", False, 0.5),
        ("2d_nan", False, [0.5, 0.5]),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_cvar(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.cvar(returns, sample_weight=sample_weight), expected
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_cvar_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.cvar(returns, sample_weight=q),
        skm.cvar(returns),
        10,
    )


@pytest.mark.parametrize("measure", [skm.value_at_risk, skm.cvar])
def test_tail_risk_2d_preserves_all_nan_columns(measure):
    returns = np.array([[np.nan, -0.2], [np.nan, 0.1]])

    result = measure(returns)

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 0.2)


def test_weighted_cvar_when_first_observation_exceeds_tail_probability():
    returns = np.array([-0.1, 0.2])
    sample_weight = np.array([0.9, 0.1])

    result = skm.cvar(returns, beta=0.95, sample_weight=sample_weight)

    np.testing.assert_allclose(result, 0.1)


@pytest.mark.parametrize(
    "returns,sample_weight,theta,beta,expected",
    [
        ("1d", False, 1.0, 0.95, 2.994984773),
        ("1d", False, 0.5, 0.5, 0.346208454),
        ("1d", True, 1.0, 0.95, 2.995102782),
        ("2d", False, 1.0, 0.95, [2.9949848, 2.9954033]),
        ("2d", True, 1.0, 0.95, [2.9951028, 2.9954657]),
        ("1d_nan", False, 1.0, 0.95, 3.080244928518),
        ("all_nan", False, 1.0, 0.95, np.nan),
        ("all_nan", True, 1.0, 0.95, np.nan),
    ],
    indirect=["returns", "sample_weight"],
)
def test_entropic_risk_measure(returns, sample_weight, theta, beta, expected):
    np.testing.assert_almost_equal(
        skm.entropic_risk_measure(
            returns, theta=theta, beta=beta, sample_weight=sample_weight
        ),
        expected,
    )


@pytest.mark.parametrize("returns", ["1d", "2d"], indirect=True)
def test_entropic_risk_measure_sample_weight(returns):
    q = np.ones(len(returns)) / len(returns)
    np.testing.assert_almost_equal(
        skm.entropic_risk_measure(returns, sample_weight=q),
        skm.entropic_risk_measure(returns),
        10,
    )


@pytest.mark.parametrize(
    "returns,expected",
    [
        ("1d", 0.213993692),
        ("1d_nan", 0.5),
        ("all_nan", np.nan),
    ],
    indirect=["returns"],
)
def test_evar(returns, expected):
    np.testing.assert_almost_equal(skm.evar(returns), expected)


@pytest.mark.parametrize(
    "returns,beta,expected",
    [
        (np.full(50, 0.01), 0.95, -0.01),
        (np.full(50, -0.01), 0.95, 0.01),
        (np.linspace(0.01, 0.1, 10), 0.95, -0.01),
        (np.linspace(1e-5, 0.1, 10), 0.95, -1e-5),
        (np.array([-0.1, -0.1, 0.0, 0.2]), 0.5, 0.1),
        (np.array([0.02, -0.01, np.nan, 0.03]), 0.0, -0.04 / 3),
        (np.array([0.02, -0.01, np.nan, 0.03]), 1.0, 0.01),
        (np.array([]), 0.95, np.nan),
    ],
)
def test_evar_closed_form(returns, beta, expected):
    np.testing.assert_almost_equal(skm.evar(returns, beta=beta), expected, 12)


@pytest.mark.parametrize("beta", [0.5, 0.9, 0.95, 0.99])
@pytest.mark.parametrize(
    "returns",
    [
        np.random.default_rng(0).uniform(0.001, 0.02, 500),
        np.random.default_rng(1).standard_t(3, 300) * 0.01,
        -np.random.default_rng(2).exponential(0.01, 200),
    ],
)
def test_evar_properties(returns, beta):
    evar = skm.evar(returns, beta=beta)
    assert skm.cvar(returns, beta=beta) <= evar <= -returns.min()

    spread = np.ptp(returns)
    thetas = np.geomspace(spread / 500, spread * 100, 2000)
    erm = min(skm.entropic_risk_measure(returns, theta=t, beta=beta) for t in thetas)
    assert evar <= erm + 1e-12
    np.testing.assert_almost_equal(evar, erm, 6)

    np.testing.assert_almost_equal(skm.evar(returns + 0.01, beta=beta), evar - 0.01, 12)
    np.testing.assert_almost_equal(skm.evar(3 * returns, beta=beta), 3 * evar, 12)


@pytest.mark.parametrize(
    "func,values",
    [
        (skm.evar, np.zeros(50)),
        (skm.edar, skm.get_drawdowns(np.full(60, 0.001))),
    ],
)
def test_evar_zero_is_positive(func, values):
    value = func(values)
    assert value == 0.0
    # A negative zero would turn the associated ratio into -inf.
    assert not np.signbit(value)


@pytest.mark.parametrize(
    "returns,expected_ndim", [("1d", 1), ("2d", 2)], indirect=["returns"]
)
@pytest.mark.parametrize("compounded", [True, False])
def test_get_cumulative_returns(returns, expected_ndim, compounded):
    res = skm.get_cumulative_returns(returns, compounded)
    assert res.ndim == expected_ndim


def test_get_cumulative_returns_nan(returns_1d_nan):
    res = skm.get_cumulative_returns(returns_1d_nan, compounded=False)
    np.testing.assert_almost_equal(res, [np.nan, 0.1, -0.4, np.nan, np.nan, -0.7, 0.1])
    res = skm.get_cumulative_returns(returns_1d_nan, compounded=True)
    np.testing.assert_almost_equal(
        res, [np.nan, 1.1, 0.55, np.nan, np.nan, 0.385, 0.693]
    )


@pytest.mark.parametrize(
    "returns,expected_ndim", [("1d", 1), ("2d", 2)], indirect=["returns"]
)
@pytest.mark.parametrize("compounded", [True, False])
def test_get_drawdowns(returns, expected_ndim, compounded):
    res = skm.get_drawdowns(returns, compounded)
    assert res.ndim == expected_ndim


@pytest.mark.parametrize(
    "compounded,expected",
    [(False, [-0.1, -0.05, -0.08]), (True, [-0.1, -0.055, -0.08335])],
)
def test_get_drawdowns_counts_loss_from_start(compounded, expected):
    # The starting wealth is the first peak, so a loss on the first observation
    # is a drawdown.
    res = skm.get_drawdowns(np.array([-0.1, 0.05, -0.03]), compounded=compounded)
    np.testing.assert_almost_equal(res, expected)


def test_get_drawdowns_nan(returns_1d_nan):
    res = skm.get_drawdowns(returns_1d_nan, compounded=False)
    np.testing.assert_almost_equal(res, [np.nan, 0.0, -0.5, np.nan, np.nan, -0.8, 0.0])
    res = skm.get_drawdowns(returns_1d_nan, compounded=True)
    np.testing.assert_almost_equal(
        res, [np.nan, 0.0, -0.5, np.nan, np.nan, -0.65, -0.37]
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.8498387),
        ("1d", True, 0.7522852),
        ("2d", False, [0.8498387, 1.5929109]),
        ("2d", True, [0.7522852, 0.9450526]),
        ("1d_nan", False, 0.8),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_drawdown_at_risk(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.drawdown_at_risk(skm.get_drawdowns(returns, compounded=compounded)),
        expected,
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 1.24805324),
        ("1d", True, 0.81809872),
        ("2d", False, [1.2480532, 2.6545675]),
        ("2d", True, [0.8180987, 0.9658947]),
        ("1d_nan", False, 0.8),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_max_drawdown(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.max_drawdown(skm.get_drawdowns(returns, compounded=compounded)), expected
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.24444925),
        ("1d", True, 0.28518241),
        ("2d", False, [0.2444493, 0.5653650]),
        ("2d", True, [0.2851824, 0.5608338]),
        ("1d_nan", False, 0.325),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_average_drawdown(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.average_drawdown(skm.get_drawdowns(returns, compounded=compounded)),
        expected,
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.92763054),
        ("1d", True, 0.78282175),
        ("2d", False, [0.9276305, 1.8762444]),
        ("2d", True, [0.7828218, 0.9536034]),
        ("1d_nan", False, 0.8),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_cdar(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.cdar(skm.get_drawdowns(returns, compounded=compounded)), expected
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.996230976),
        ("1d", True, 0.791260923),
        ("1d_nan", False, 0.8),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_edar(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.edar(skm.get_drawdowns(returns, compounded=compounded)), expected
    )


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.360642004),
        ("1d", True, 0.383078682),
        ("2d", False, [0.360642, 0.7670837]),
        ("2d", True, [0.3830787, 0.6368838]),
        ("1d_nan", False, 0.47169905),
        ("all_nan", False, np.nan),
        ("all_nan", True, np.nan),
    ],
    indirect=["returns"],
)
def test_ulcer_index(returns, compounded, expected):
    np.testing.assert_almost_equal(
        skm.ulcer_index(skm.get_drawdowns(returns, compounded=compounded)), expected
    )


@pytest.mark.parametrize("n_observations", [10, 150, 3520])
def test_owa_gmd_weights(n_observations):
    res = skm.owa_gmd_weights(n_observations)
    assert res.shape == (n_observations,)


@pytest.mark.parametrize(
    "returns,expected",
    [
        ("1d", 0.0278023),
        ("2d", [0.0278023, 0.0400147]),
        ("1d_nan", 0.7166666),
        ("all_nan", np.nan),
    ],
    indirect=["returns"],
)
def test_gini_mean_difference(returns, expected):
    np.testing.assert_almost_equal(skm.gini_mean_difference(returns), expected)


def test_gini_mean_difference_2d_handles_nan_columns_independently():
    returns = np.array(
        [
            [1.0, np.nan, 2.0],
            [3.0, np.nan, 4.0],
            [np.nan, np.nan, 6.0],
        ]
    )

    result = skm.gini_mean_difference(returns)

    np.testing.assert_allclose(
        result[[0, 2]],
        [
            skm.gini_mean_difference(returns[:, 0]),
            skm.gini_mean_difference(returns[:, 2]),
        ],
    )
    assert np.isnan(result[1])


@pytest.mark.parametrize(
    "sample_weight,expected", [(True, 6216.7875118)], indirect=["sample_weight"]
)
def test_effective_number_assets(sample_weight, expected):
    np.testing.assert_almost_equal(skm.effective_number_assets(sample_weight), expected)


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("2d", False, [[1.0, 0.32859534], [0.32859534, 1.0]]),
        ("2d", True, [[1.0, 0.328578], [0.328578, 1.0]]),
    ],
    indirect=["returns", "sample_weight"],
)
def test_correlation(returns, sample_weight, expected):
    np.testing.assert_almost_equal(
        skm.correlation(returns, sample_weight=sample_weight), expected
    )
