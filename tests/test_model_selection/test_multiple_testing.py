import numpy as np
import pytest

from skfolio.model_selection import MultipleTestingResult, multiple_testing_hurdle
from skfolio.model_selection._multiple_testing import (
    _draw_indices,
    _exceedance_counts,
    _resampled_t_statistics,
)

# Small enough to keep the suite fast; the paper uses 100 x 100.
FAST = {"n_perturbations": 15, "n_simulations": 15, "random_state": 3}


@pytest.fixture(scope="module")
def X():
    """36 null trials and 4 with a genuine edge, sharing a time index."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 40))
    x[:, -4:] += 4.0 / np.sqrt(200)
    return x


def test_returns_result(X):
    result = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    assert isinstance(result, MultipleTestingResult)
    assert result.n_trials == 40
    assert result.n_observations == 200
    assert result.n_non_null == 4
    assert result.hurdles.shape == result.false_discovery_rates.shape
    assert set(result.summary().index) >= {"Hurdle", "Type I error (FDR)"}


# ---------------------------------------------------------------------------
# The counts shortcut must agree with the obvious implementation it replaces.
# ---------------------------------------------------------------------------
def test_resampled_t_statistics_match_direct_indexing():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 6))
    indices = rng.integers(0, 50, size=(7, 50))

    from_counts = _resampled_t_statistics(values, values**2, indices)

    expected = np.empty((7, 6))
    for i, index in enumerate(indices):
        sample = values[index]
        mean = sample.mean(axis=0)
        expected[i] = mean / (sample.std(axis=0, ddof=1) / np.sqrt(50))

    np.testing.assert_allclose(from_counts, expected, rtol=1e-9, atol=1e-9)


def test_exceedance_counts_match_direct_comparison():
    rng = np.random.default_rng(1)
    absolute_t = np.abs(rng.normal(size=(5, 12))) * 2
    hurdles = np.arange(0.0, 6.0 + 1e-9, 0.05)

    counts = _exceedance_counts(absolute_t, hurdles, hurdles.size)
    expected = (absolute_t[:, None, :] >= hurdles[None, :, None]).sum(axis=2)

    np.testing.assert_array_equal(counts, expected)


def test_exceedance_counts_handles_no_non_null_trials():
    hurdles = np.arange(0.0, 1.0, 0.05)
    counts = _exceedance_counts(np.empty((4, 0)), hurdles, hurdles.size)
    assert counts.shape == (4, hurdles.size)
    assert not counts.any()


# ---------------------------------------------------------------------------
# Two values fixed by construction, so they never pass through the resampling.
# A resampler that silently did nothing would still have to reproduce them.
# ---------------------------------------------------------------------------
def test_error_rates_at_a_zero_hurdle_are_analytic(X):
    """Every trial is declared significant, so the table is fixed whatever the data did.

    TP = n_non_null, FP = n_trials - n_non_null, FN = TN = 0, giving a false
    discovery rate of exactly 1 - non_null_ratio, a miss rate of exactly zero
    (empty denominator) and an odds ratio of exactly zero (no misses).
    """
    result = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    assert result.hurdles[0] == 0.0
    assert result.false_discovery_rates[0] == pytest.approx(36 / 40, abs=1e-12)
    assert result.miss_rates[0] == 0.0
    assert result.odds_ratios[0] == 0.0


def test_error_rates_at_an_unreachable_hurdle_are_analytic(X):
    """Nothing is declared significant, so FP = TP = 0 and FN = n_non_null exactly."""
    result = multiple_testing_hurdle(
        X, non_null_ratio=0.10, hurdles=[0.0, 99.0], **FAST
    )
    assert result.false_discovery_rates[-1] == 0.0
    assert result.odds_ratios[-1] == 0.0
    assert result.miss_rates[-1] == pytest.approx(4 / 40, abs=1e-12)


# ---------------------------------------------------------------------------
# Directions the definitions force
# ---------------------------------------------------------------------------
def test_type_1_falls_and_type_2_rises_with_the_hurdle(X):
    result = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    assert result.false_discovery_rates[0] > result.false_discovery_rates[-1]
    assert result.miss_rates[0] < result.miss_rates[-1]


def test_hurdle_is_non_increasing_in_the_assumed_non_null_ratio(X):
    """More non-null trials means more true positives at any hurdle, so the target is
    met sooner and the hurdle cannot rise."""
    hurdles = [
        multiple_testing_hurdle(X, non_null_ratio=p, **FAST).hurdle
        for p in (0.0, 0.05, 0.10, 0.20, 0.40)
    ]
    assert hurdles == sorted(hurdles, reverse=True), hurdles


def test_a_costlier_false_discovery_buys_a_stricter_hurdle(X):
    """Targeting an odds ratio of 1/k prices a false discovery at k times a miss."""
    hurdles = [
        multiple_testing_hurdle(
            X, non_null_ratio=0.10, target=1 / k, criterion="odds_ratio", **FAST
        ).hurdle
        for k in (1, 3, 10)
    ]
    assert hurdles == sorted(hurdles), hurdles
    assert hurdles[0] < hurdles[-1]


def test_planted_trials_are_the_ones_selected(X):
    result = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    planted = set(range(36, 40))
    selected = set(result.selected.tolist())
    assert len(selected & planted) >= 3
    assert not selected - planted


def test_a_pure_noise_panel_selects_almost_nothing():
    rng = np.random.default_rng(5)
    result = multiple_testing_hurdle(
        rng.normal(size=(200, 40)), non_null_ratio=0.10, **FAST
    )
    assert len(result.selected) <= 1


# ---------------------------------------------------------------------------
# Degenerate assumption, reproducibility, block bootstrap
# ---------------------------------------------------------------------------
def test_a_zero_non_null_ratio_leaves_type_2_degenerate(X):
    """With no non-null trials a miss is impossible, so a zero miss rate says nothing
    about the hurdle; the Type I rate becomes the family-wise error rate."""
    result = multiple_testing_hurdle(X, non_null_ratio=0.0, **FAST)
    assert result.n_non_null == 0
    assert result.miss_rate == 0.0
    assert np.all(result.miss_rates == 0.0)
    assert np.all(result.odds_ratios == 0.0)


def test_reproducible_given_random_state(X):
    a = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    b = multiple_testing_hurdle(X, non_null_ratio=0.10, **FAST)
    assert a.hurdle == b.hurdle
    np.testing.assert_array_equal(a.miss_rates, b.miss_rates)


def test_different_random_states_are_not_identical(X):
    a = multiple_testing_hurdle(
        X, non_null_ratio=0.10, n_perturbations=15, n_simulations=15, random_state=1
    )
    b = multiple_testing_hurdle(
        X, non_null_ratio=0.10, n_perturbations=15, n_simulations=15, random_state=2
    )
    assert not np.array_equal(a.false_discovery_rates, b.false_discovery_rates)


def test_block_bootstrap_draws_valid_indices():
    rng = np.random.RandomState(0)
    indices = _draw_indices(rng, 50, 8, block_size=5.0)
    assert indices.shape == (8, 50)
    assert indices.min() >= 0
    assert indices.max() < 50
    # Blocks make consecutive draws follow one another far more often than chance.
    consecutive = (np.diff(indices, axis=1) == 1).mean()
    assert consecutive > 0.5


def test_block_bootstrap_runs_end_to_end(X):
    result = multiple_testing_hurdle(X, non_null_ratio=0.10, block_size=5.0, **FAST)
    assert result.hurdle > 0
    assert result.false_discovery_rates[0] == pytest.approx(36 / 40, abs=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("non_null_ratio", [-0.01, 1.0, 1.5])
def test_invalid_non_null_ratio_raises(X, non_null_ratio):
    with pytest.raises(ValueError, match="non_null_ratio"):
        multiple_testing_hurdle(X, non_null_ratio=non_null_ratio, **FAST)


def test_invalid_criterion_raises(X):
    with pytest.raises(ValueError, match="criterion"):
        multiple_testing_hurdle(X, non_null_ratio=0.1, criterion="wrong", **FAST)


@pytest.mark.parametrize("target", [0.0, -1.0])
def test_invalid_target_raises(X, target):
    with pytest.raises(ValueError, match="target"):
        multiple_testing_hurdle(X, non_null_ratio=0.1, target=target, **FAST)


@pytest.mark.parametrize(
    "hurdles",
    [[0.5, 1.0], [0.0, 1.0, 0.5], np.empty(0)],
    ids=["not-zero", "unsorted", "empty"],
)
def test_invalid_hurdles_raise(X, hurdles):
    with pytest.raises(ValueError, match="hurdles"):
        multiple_testing_hurdle(X, non_null_ratio=0.1, hurdles=hurdles, **FAST)


@pytest.mark.parametrize("kwargs", [{"n_perturbations": 0}, {"n_simulations": 0}])
def test_invalid_iteration_counts_raise(X, kwargs):
    with pytest.raises(ValueError, match="strictly positive"):
        multiple_testing_hurdle(X, non_null_ratio=0.1, random_state=3, **kwargs)


def test_too_few_observations_raises():
    with pytest.raises(ValueError):
        multiple_testing_hurdle(np.ones((2, 4)), non_null_ratio=0.1, **FAST)
