"""Test Measure module."""

import numpy as np
import pytest

import skfolio.measures as skm
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns


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


@pytest.fixture(
    scope="module",
    params=["1d", "2d"],
)
def returns(request, returns_1d, returns_2d):
    if request.param == "1d":
        return returns_1d
    elif request.param == "2d":
        return returns_2d
    else:
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


@pytest.mark.parametrize(
    "returns,sample_weight,expected",
    [
        ("1d", False, -8.0259e-06),
        ("1d", True, -1.04409e-05),
        ("2d", False, [-8.0259e-06, 2.02371e-05]),
        ("2d", True, [-1.04409e-05, 2.18454e-05]),
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
    "returns,sample_weight,expected",
    [
        ("1d", False, 0.059240073),
        ("1d", True, 0.058665975),
        ("2d", False, [0.0592401, 0.0852369]),
        ("2d", True, [0.0586659, 0.0852807]),
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


@pytest.mark.parametrize(
    "returns,sample_weight,theta,beta,expected",
    [
        ("1d", False, 1.0, 0.95, 2.994984773),
        ("1d", False, 0.5, 0.5, 0.346208454),
        ("1d", True, 1.0, 0.95, 2.995102782),
        ("2d", False, 1.0, 0.95, [2.9949848, 2.9954033]),
        ("2d", True, 1.0, 0.95, [2.9951028, 2.9954657]),
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
    "returns,expected", [("1d", 0.213993692)], indirect=["returns"]
)
def test_evar(returns, expected):
    np.testing.assert_almost_equal(skm.evar(returns), expected)


@pytest.mark.parametrize(
    "returns,expected_ndim", [("1d", 1), ("2d", 2)], indirect=["returns"]
)
@pytest.mark.parametrize("compounded", [True, False])
def test_get_cumulative_returns(returns, expected_ndim, compounded):
    res = skm.get_cumulative_returns(returns, compounded)
    assert res.ndim == expected_ndim


@pytest.mark.parametrize(
    "returns,expected_ndim", [("1d", 1), ("2d", 2)], indirect=["returns"]
)
@pytest.mark.parametrize("compounded", [True, False])
def test_get_drawdowns(returns, expected_ndim, compounded):
    res = skm.get_drawdowns(returns, compounded)
    assert res.ndim == expected_ndim


@pytest.mark.parametrize(
    "returns,compounded,expected",
    [
        ("1d", False, 0.8498387),
        ("1d", True, 0.7522852),
        ("2d", False, [0.8498387, 1.5929109]),
        ("2d", True, [0.7522852, 0.9450526]),
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
        ("2d", False, [0.2444493, 0.5652300]),
        ("2d", True, [0.2851824, 0.5607065]),
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
        ("2d", False, [0.360642, 0.7670693]),
        ("2d", True, [0.3830787, 0.6368674]),
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
    ],
    indirect=["returns"],
)
def test_gini_mean_difference(returns, expected):
    np.testing.assert_almost_equal(skm.gini_mean_difference(returns), expected)


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
