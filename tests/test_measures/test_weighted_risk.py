"""Independent references and validation for observation-weighted measures."""

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import skfolio.measures as mt

WEIGHTED_MEASURES = [
    mt.mean,
    mt.mean_absolute_deviation,
    mt.first_lower_partial_moment,
    mt.variance,
    mt.semi_variance,
    mt.standard_deviation,
    mt.semi_deviation,
    mt.third_central_moment,
    mt.skew,
    mt.fourth_central_moment,
    mt.kurtosis,
    mt.fourth_lower_partial_moment,
    mt.value_at_risk,
    mt.cvar,
    mt.entropic_risk_measure,
    mt.evar,
    mt.drawdown_at_risk,
    mt.average_drawdown,
    mt.cdar,
    mt.edar,
    mt.ulcer_index,
]


@pytest.mark.parametrize("measure", [mt.cvar, mt.cdar])
@pytest.mark.parametrize("sample_weight", [None, [1, 1, 1, 1], [0, 0.3, 0.7, 0]])
@pytest.mark.parametrize("matrix", [False, True])
def test_conditional_tail_at_confidence_one(measure, sample_weight, matrix):
    values = np.array([-0.8, -0.2, -0.1, np.nan])
    if matrix:
        values = np.column_stack([values, [np.nan, -0.1, -0.4, -0.3]])
    if sample_weight is None or sample_weight[0] > 0:
        expected = [0.8, 0.4] if matrix else 0.8
    else:
        expected = [0.2, 0.4] if matrix else 0.2
    np.testing.assert_allclose(
        measure(values, beta=1, sample_weight=sample_weight), expected
    )
    np.testing.assert_allclose(
        measure(values, beta=np.nextafter(1.0, 0.0), sample_weight=sample_weight),
        expected,
    )
    assert np.isnan(
        measure([], beta=1, sample_weight=None if sample_weight is None else [])
    )
    assert np.isnan(measure([np.nan], beta=1, sample_weight=[1]))
    assert np.isnan(measure([-0.2], beta=1, sample_weight=[0]))


@pytest.mark.parametrize("measure", [mt.skew, mt.kurtosis])
@pytest.mark.parametrize(
    "values",
    [
        [0.5],
        [0.5, 0.5],
        [np.nan, 0.5, 0.5],
        [0.1] * 7,
        [1 / 3] * 11,
        [0.01] * 3,
        [0.1] * 30,
        [0.01] * 252,
        [1 / 3] * 1000,
    ],
)
@pytest.mark.parametrize("weighting", ["none", "uniform", "nonuniform"])
@pytest.mark.parametrize("matrix", [False, True])
def test_undefined_standardized_moments_are_quiet(measure, values, weighting, matrix):
    sample_weight = None
    if weighting == "uniform":
        sample_weight = np.ones(len(values))
    elif weighting == "nonuniform":
        sample_weight = np.arange(1.0, len(values) + 1)
    if matrix:
        values = np.column_stack([values, values])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = measure(values, sample_weight=sample_weight)
    assert np.isnan(result).all()
    assert np.shape(result) == ((2,) if matrix else ())


@pytest.mark.parametrize("measure", [mt.skew, mt.kurtosis])
@pytest.mark.parametrize("weighted", [False, True])
def test_standardized_moment_threshold_is_column_specific(measure, weighted):
    returns = np.column_stack(
        [np.full(7, 0.1), [-0.03, 0.04, np.nan, 0.02, 0.06, -0.01, 0.04]]
    )
    returns[2, 0] = np.nan
    sample_weight = np.array([0, 1, 2, 3, 4, 5, 6]) if weighted else None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = measure(returns, sample_weight=sample_weight)
    assert np.isnan(result[0])
    values = returns[:, 1]
    if weighted:
        values = np.repeat(values, sample_weight)
    expected = (
        stats.skew(values, nan_policy="omit")
        if measure is mt.skew
        else stats.kurtosis(values, fisher=False, nan_policy="omit")
    )
    np.testing.assert_allclose(result[1], expected, rtol=1e-12)


@pytest.mark.parametrize("measure, expected", [(mt.skew, 0), (mt.kurtosis, 1.5)])
def test_standardized_moments_preserve_resolvable_variation(measure, expected):
    eps = np.finfo(float).eps
    assert np.isnan(measure([1 - eps, 1, 1 + eps]))
    np.testing.assert_allclose(measure([1 - 4 * eps, 1, 1 + 4 * eps]), expected)
    np.testing.assert_allclose(measure([-1e-20, 0, 1e-20]), expected)


@pytest.mark.parametrize(
    "measure, expected", [(mt.skew, 1 / np.sqrt(2)), (mt.kurtosis, 1.5)]
)
@pytest.mark.parametrize("sample_weight", [None, [1, 1, 1]])
def test_standardized_moments_correct_rounding_in_centering(
    measure, expected, sample_weight
):
    # The represented returns differ, but their exact mean is not representable.
    returns = 1 + np.finfo(float).eps * np.array([0, 0, 16])
    np.testing.assert_allclose(measure(returns, sample_weight=sample_weight), expected)


@pytest.mark.parametrize("measure", [*WEIGHTED_MEASURES, mt.correlation])
@pytest.mark.parametrize(
    "weights",
    [
        [1],
        [[0.2, 0.3, 0.5]],
        [0.2, -0.1, 0.9],
        [0.2, np.nan, 0.8],
        [0.2, np.inf, 0.8],
        [0.2, -np.inf, 0.8],
        [0.2, 0.3j, 0.8],
        ["bad", 0, 1],
    ],
)
def test_invalid_sample_weight(measure, weights):
    returns = np.array([0.1, np.nan, -0.2])
    if measure is mt.correlation:
        returns = returns[:, None]
    with pytest.raises(ValueError, match="sample_weight"):
        measure(returns, sample_weight=weights)


@pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
@pytest.mark.parametrize("scale", [1e-310, 4e307])
@pytest.mark.parametrize("missing", [False, True])
def test_extreme_relative_weights(measure, scale, missing):
    returns = np.array([0.1, np.nan if missing else 0.2, -0.2, 0.3])
    weights = np.array([1.0, 2.0, 1.0, 2.0])
    expected = measure(returns, sample_weight=weights)
    np.testing.assert_allclose(
        measure(returns, sample_weight=weights * scale),
        expected,
        rtol=1e-10,
        atol=1e-14,
    )


@pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
def test_normalize_after_excluding_missing_returns(measure):
    # Normalizing first would erase both usable weights through underflow.
    returns = np.array([np.nan, -0.2, 0.1])
    weights = np.array([1e300, 1e-300, 2e-300])
    np.testing.assert_allclose(
        measure(returns, sample_weight=weights),
        measure(returns[1:], sample_weight=[1, 2]),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "measure", [mt.mean, mt.variance, mt.fourth_lower_partial_moment]
)
def test_extreme_weights_normalize_separately_for_each_column(measure):
    returns = np.array(
        [[np.nan, 0.3, np.nan], [-0.2, np.nan, np.nan], [0.1, 0.1, np.nan]]
    )
    weights = np.array([1e300, 1e-300, 2e-300])
    actual = measure(returns, sample_weight=weights)
    np.testing.assert_allclose(
        actual[0], measure(returns[1:, 0], sample_weight=[1, 2]), atol=1e-14
    )
    np.testing.assert_allclose(
        actual[1],
        measure(returns[[0, 2], 1], sample_weight=weights[[0, 2]]),
        atol=1e-14,
    )
    assert np.isnan(actual[2])


@pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
def test_weighted_inputs_are_not_modified(measure):
    returns = np.array([0.1, np.nan, -0.2, 0.3])[::-1]
    weights = np.array([1.0, 2.0, 0.0, 4.0])[::-1]
    expected_returns, expected_weights = returns.copy(), weights.copy()
    returns.setflags(write=False)
    weights.setflags(write=False)
    measure(returns, sample_weight=weights)
    np.testing.assert_array_equal(returns, expected_returns)
    np.testing.assert_array_equal(weights, expected_weights)


@pytest.mark.parametrize("measure", [mt.evar, mt.edar])
@pytest.mark.parametrize("beta", [0.0, 0.3, 0.95, 1.0])
def test_weighted_evar_uniform_missing_and_zero_mass(measure, beta):
    returns = np.array([0.1, np.nan, -0.2, 0.3])
    np.testing.assert_allclose(
        measure(returns, beta=beta, sample_weight=np.ones(4)),
        measure(returns, beta=beta),
        atol=1e-12,
    )
    assert np.isnan(measure(returns, beta=beta, sample_weight=np.zeros(4)))
    assert np.isnan(measure([], beta=beta, sample_weight=np.array([])))
    assert np.isnan(measure([np.nan], beta=beta, sample_weight=np.ones(1)))


@pytest.mark.parametrize("target", [None, 0.01, np.array([0.01, -0.02])])
def test_weighted_fourth_lower_partial_moment_reference(target):
    returns = np.array([[0.1, -0.2], [np.nan, -0.05], [-0.2, np.nan], [0.3, 0.2]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    expected = []
    for column_index, column in enumerate(returns.T):
        valid = ~np.isnan(column)
        probabilities = weights[valid] / weights[valid].sum()
        center = (
            probabilities @ column[valid]
            if target is None
            else np.broadcast_to(target, 2)[column_index]
        )
        expected.append(probabilities @ np.maximum(center - column[valid], 0) ** 4)
    np.testing.assert_allclose(
        mt.fourth_lower_partial_moment(returns, target, weights), expected
    )


@pytest.mark.parametrize(
    "losses,weights,beta",
    [
        ([1.0, 0.0], [1e-150, 1.0], 0.95),
        ([-1.0, -1e-6, 0.0], [0.01, 0.99, 1e-300], 0.95),
        ([1.0, 0.0], [1e-300, 1e300], 0.95),
        ([0.2, 0.2, 0.01, -0.1], [0.01, 0.01, 0.28, 0.7], 0.8),
    ],
)
def test_weighted_evar_independent_minimization(losses, weights, beta):
    losses, weights = np.array(losses), np.array(weights)
    log_weights = np.log(weights) - logsumexp(np.log(weights))

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return temperature * (
            logsumexp(log_weights + losses / temperature) - np.log1p(-beta)
        )

    reference = minimize_scalar(
        objective, bounds=(-30, 10), method="bounded", options={"xatol": 1e-13}
    )
    assert reference.success
    actual = mt.evar(-losses, beta=beta, sample_weight=weights)
    np.testing.assert_allclose(actual, reference.fun, rtol=1e-7, atol=1e-13)
    np.testing.assert_allclose(
        mt.evar(-3 * losses + 0.4, beta=beta, sample_weight=weights),
        3 * actual - 0.4,
        atol=1e-12,
    )


def test_weighted_evar_support_and_endpoints():
    returns = np.array([-0.9, -0.2, -0.2, 0.1])
    weights = np.array([0.0, 0.02, 0.03, 0.95])
    assert mt.evar(returns, beta=1, sample_weight=weights) == 0.2
    np.testing.assert_allclose(
        mt.evar(returns, beta=0, sample_weight=weights), -(weights @ returns)
    )
    assert mt.evar(returns, beta=0.96, sample_weight=weights) == 0.2
    assert mt.evar([-0.2, -0.2], sample_weight=[0.1, 0.9]) == 0.2
    assert mt.evar(returns, sample_weight=[0, 0, 0, 1]) == -0.1


@pytest.mark.parametrize("beta", [1e-16, 1e-12])
@pytest.mark.parametrize("probability", [0.01, 0.2, 0.7])
@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_weighted_evar_small_confidence_level(beta, probability, scale):
    # For a Bernoulli loss, EVaR approaches mean + sqrt(2 * variance * -log(1-beta)).
    # The omitted terms are of order beta, below the absolute tolerance here.
    expected = probability + np.sqrt(
        -2 * np.log1p(-beta) * probability * (1 - probability)
    )
    actual = mt.evar(
        [-1.0, 0.0],
        beta=beta,
        sample_weight=scale * np.array([probability, 1 - probability]),
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_weighted_evar_near_zero_confidence_matches_mean():
    returns = np.array(
        [0.10476804354061553, -0.0007600459443068436, 0.2159497441431377]
    )
    weights = np.array(
        [1.0350724827843268e-115, 6.404277376335064e-240, 1.4088183316765047e-118]
    )
    np.testing.assert_allclose(
        mt.evar(returns, beta=1e-30, sample_weight=weights),
        -np.average(returns, weights=weights),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("measure", [mt.skew, mt.kurtosis])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_shape_statistics_with_one_observation(measure, dtype):
    with np.errstate(invalid="ignore"):
        value = measure(np.array([0.123], dtype=dtype), sample_weight=[0.37])
    assert np.isnan(value)


@pytest.mark.parametrize("compounded", [False, True])
def test_weighted_drawdowns_keep_zero_weight_returns(compounded):
    returns = np.array([-0.2, np.nan, 0.1, 0.1])
    weights = np.array([0.0, 0.0, 0.5, 0.5])
    drawdowns = mt.get_drawdowns(returns, compounded=compounded)
    magnitudes = np.array([0.12, 0.032]) if compounded else np.array([0.1, 0.0])
    np.testing.assert_allclose(
        mt.average_drawdown(drawdowns, weights), magnitudes.mean(), atol=1e-15
    )
    np.testing.assert_allclose(
        mt.ulcer_index(drawdowns, weights), np.sqrt(np.mean(magnitudes**2)), atol=1e-15
    )
    np.testing.assert_allclose(
        mt.cdar(drawdowns, beta=0, sample_weight=weights), magnitudes.mean(), atol=1e-15
    )
    np.testing.assert_allclose(
        mt.cdar(drawdowns, beta=0.5, sample_weight=weights), magnitudes.max()
    )
    assert mt.drawdown_at_risk(
        drawdowns, beta=1, sample_weight=weights
    ) < mt.max_drawdown(drawdowns)
    np.testing.assert_allclose(mt.max_drawdown(drawdowns), 0.2)


def test_weighted_drawdown_risk_ordering_and_fractional_tail():
    drawdowns = np.array([-0.4, -0.2, -0.1, 0.0])
    weights = np.array([0.0, 0.1, 0.4, 0.5])
    value_at_risk = mt.drawdown_at_risk(drawdowns, beta=0.8, sample_weight=weights)
    conditional = mt.cdar(drawdowns, beta=0.8, sample_weight=weights)
    entropic = mt.edar(drawdowns, beta=0.8, sample_weight=weights)
    np.testing.assert_allclose(conditional, 0.15)
    assert value_at_risk <= conditional <= entropic <= 0.2
