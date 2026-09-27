"""Contract checks for the discussion prototype, not a full estimator suite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from skfolio.model_selection import OnlineGridSearch, online_predict
from skfolio.optimization import ExponentiatedGradient


@pytest.fixture
def X():
    """Provide deterministic finite simple returns."""
    return np.random.default_rng(17).normal(0, 0.02, size=(80, 4))


def test_textbook_update():
    """Compare one update with the direct multiplicative formula."""
    initial = np.array([0.3, 0.7])
    relatives = np.array([1.10, 0.95])
    expected = initial * np.exp(0.2 * relatives / (initial @ relatives))
    expected /= expected.sum()
    model = ExponentiatedGradient(learning_rate=0.2, initial_weights=initial)
    model.partial_fit((relatives - 1)[None, :])
    np.testing.assert_allclose(model.weights_, expected)
    np.testing.assert_array_equal(initial, [0.3, 0.7])


def test_block_row_and_chunk_equivalence(X):
    """Partitioning observations must not change learning."""
    batch = ExponentiatedGradient().fit(X)
    rows = ExponentiatedGradient()
    chunks = ExponentiatedGradient()
    for row in X:
        rows.partial_fit(row[None, :])
    for chunk in np.array_split(X, 7):
        chunks.partial_fit(chunk)
    for model in (rows, chunks):
        np.testing.assert_allclose(model.weights_, batch.weights_)
        assert model.n_observations_ == len(X)
        assert (model.weights_ >= 0).all()
        assert model.weights_.sum() == pytest.approx(1)


def test_fit_reset_and_clone(X):
    """Refitting resets state and cloning retains only parameters."""
    model = ExponentiatedGradient().partial_fit(X[:40])
    model.fit(X[40:])
    np.testing.assert_allclose(
        model.weights_, ExponentiatedGradient().fit(X[40:]).weights_
    )
    assert model.n_observations_ == 40
    assert not hasattr(clone(model), "weights_")


def test_execution_holdings_do_not_replace_algorithm_target(X):
    """Execution metadata must not alter the EG update reference."""
    model = ExponentiatedGradient().fit(X[:40])
    model.set_params(previous_weights=np.array([0.7, 0.1, 0.1, 0.1]))
    model.partial_fit(X[40:])
    np.testing.assert_allclose(model.weights_, ExponentiatedGradient().fit(X).weights_)


def test_online_evaluation_is_next_period_and_causal(X):
    """Verify predict-before-update timing and execution-state propagation."""
    estimator = ExponentiatedGradient(portfolio_params={"weight_drift": True})
    result = online_predict(estimator, X, warmup_size=20, test_size=1)
    manual = clone(estimator).fit(X[:20])
    expected = []
    for row in X[20:]:
        expected.append(manual.weights_ @ row)
        manual.partial_fit(row[None, :])
    np.testing.assert_allclose(result.returns, expected, atol=1e-14)
    changed = X.copy()
    changed[50:] = 0.25
    other = online_predict(estimator, changed, warmup_size=20, test_size=1)
    np.testing.assert_allclose(result.returns[:30], other.returns[:30])
    assert not hasattr(estimator, "weights_")  # evaluator clones the input
    for previous, current in zip(
        result.portfolios[:-1], result.portfolios[1:], strict=True
    ):
        np.testing.assert_allclose(current.previous_weights, previous.ending_weights)


def test_feature_order_is_checked(X):
    """Reject reordered named assets before updating learning."""
    frame = pd.DataFrame(X, columns=list("abcd"))
    model = ExponentiatedGradient().fit(frame[:40])
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(frame[40:][list("dcba")])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, -2.0])
def test_invalid_block_does_not_advance_learning(X, bad):
    """Validate the full input block before consuming its first row."""
    model = ExponentiatedGradient().fit(X[:40])
    old_weights = model.weights_.copy()
    block = X[40:].copy()
    block[-1, 0] = bad
    with pytest.raises(ValueError):
        model.partial_fit(block)
    assert model.n_observations_ == 40
    np.testing.assert_array_equal(model.weights_, old_weights)


@pytest.mark.parametrize("rate", [-0.1, np.inf, np.nan, True, "bad"])
def test_invalid_learning_rate(X, rate):
    """Reject unsupported learning-rate values."""
    with pytest.raises(ValueError, match="learning_rate"):
        ExponentiatedGradient(learning_rate=rate).fit(X)


def test_zero_rate_keeps_initial_target(X):
    """Zero learning rate is a constant-target baseline."""
    model = ExponentiatedGradient(learning_rate=0).fit(X)
    np.testing.assert_allclose(model.weights_, model.initial_weights_)


def test_schedule_sees_observations_not_blocks(X):
    """A schedule must advance once per observation, including warmup."""

    def schedule(t):
        return 0.1 / np.sqrt(t + 1)

    batch = ExponentiatedGradient(learning_rate=schedule).fit(X)
    chunks = ExponentiatedGradient(learning_rate=schedule)
    for chunk in np.array_split(X, 5):
        chunks.partial_fit(chunk)
    np.testing.assert_allclose(batch.weights_, chunks.weights_)


def test_bounds_with_named_assets_and_reset(X):
    """Reuse skfolio dictionary handling and keep initial and updated targets feasible."""
    frame = pd.DataFrame(X, columns=list("abcd"))
    model = ExponentiatedGradient(min_weights={"a": 0.4}, max_weights={"d": 0.1}).fit(
        frame
    )
    for weights in (model.initial_weights_, model.weights_):
        assert weights[0] >= 0.4 - 1e-12
        assert weights[3] <= 0.1 + 1e-12
        assert weights.sum() == pytest.approx(1)
    model.set_params(min_weights=0, max_weights=1).fit(frame)
    np.testing.assert_allclose(
        model.weights_, ExponentiatedGradient().fit(frame).weights_
    )


@pytest.mark.parametrize(
    "bounds",
    [
        {"min_weights": 0.3},
        {"max_weights": 0.2},
        {"min_weights": -0.1},
        {"max_weights": np.nan},
    ],
)
def test_infeasible_bounds(X, bounds):
    """Reject inconsistent or unsupported bounds before updating."""
    with pytest.raises(ValueError):
        ExponentiatedGradient(**bounds).fit(X)


def test_kl_projection_matches_independent_convex_solution():
    """Check the constrained optimum, not just whether weights satisfy bounds."""
    import cvxpy as cp

    from skfolio.optimization.online._eg_update import project_entropy

    q = np.array([0.7, 0.2, 0.08, 0.02])
    lower = np.array([0, 0, 0.2, 0.1])
    upper = np.array([0.4, 0.6, 0.5, 0.4])
    actual = project_entropy(np.log(q), lower, upper)
    w = cp.Variable(4)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.kl_div(w, q))), [cp.sum(w) == 1, w >= lower, w <= upper]
    )
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10)
    np.testing.assert_allclose(actual, w.value, atol=1e-6)


def test_fixed_and_excluded_allocations(X):
    """Handle boundary feasible sets without a root-finding failure."""
    fixed = [0.5, 0.5, 0, 0]
    model = ExponentiatedGradient(min_weights=fixed, max_weights=fixed).fit(X)
    np.testing.assert_allclose(model.weights_, fixed)


def test_failed_schedule_does_not_commit_observation(X):
    """An invalid update leaves the previous learning state intact."""
    model = ExponentiatedGradient().fit(X[:40])
    before = model.weights_.copy()
    model.set_params(learning_rate=lambda t: np.nan)
    with pytest.raises(ValueError, match="learning_rate"):
        model.partial_fit(X[40:41])
    np.testing.assert_array_equal(before, model.weights_)
    assert model.n_observations_ == 40


def test_online_grid_search_with_bounds(X):
    """Exercise cloning, parameter selection and final incremental refitting."""
    search = OnlineGridSearch(
        ExponentiatedGradient(max_weights=0.4),
        {"learning_rate": [0.01, 0.1]},
        warmup_size=20,
        test_size=1,
        n_jobs=1,
        error_score="raise",
    ).fit(X)
    assert search.best_estimator_.n_observations_ == len(X)
    assert np.all(search.best_estimator_.weights_ <= 0.4 + 1e-12)


def test_changed_bounds_require_refit(X):
    """Changing constraints cannot silently leave an old feasible set active."""
    model = ExponentiatedGradient().fit(X[:40])
    model.set_params(max_weights=0.4)
    with pytest.raises(ValueError, match="call fit"):
        model.partial_fit(X[40:])
    assert model.n_observations_ == 40
    model.fit(X)
    assert np.all(model.weights_ <= 0.4 + 1e-12)


@pytest.mark.parametrize("test_size", [2, 5])
def test_rejects_multi_period_evaluation(X, test_size):
    """Intermediate unexecuted targets must not silently become a backtest."""
    with pytest.raises(ValueError, match="one observation"):
        online_predict(ExponentiatedGradient(), X, warmup_size=20, test_size=test_size)


def test_search_rejects_multi_period_evaluation(X):
    """Hyperparameter search must obey the same one-period contract."""
    with pytest.raises(ValueError, match="one observation"):
        OnlineGridSearch(
            ExponentiatedGradient(),
            {"learning_rate": [0.05]},
            warmup_size=20,
            test_size=3,
            n_jobs=1,
            error_score="raise",
        ).fit(X)


@pytest.mark.parametrize("initial", [[0, 0, 0, 1], [1, 1, 1, 1], [0.5, 0.5]])
def test_invalid_initial_allocation(X, initial):
    """The reference allocation must be positive and match the universe."""
    with pytest.raises(ValueError, match="initial_weights"):
        ExponentiatedGradient(initial_weights=initial).fit(X)
