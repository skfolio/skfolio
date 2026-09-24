from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.pre_selection import SelectNonDominated


def test_select_non_dominated(X):
    with config_context(transform_output="pandas"):
        model = SelectNonDominated(min_n_assets=10)
        model.fit(X)
        assert np.all(
            model.feature_names_in_[model.to_keep_] == model.get_feature_names_out()
        )
        new_X = model.transform(X)
        assert new_X.shape[0] == X.shape[0]
        assert new_X.shape[1] < X.shape[0]
        assert new_X.shape[1] >= 10
        assert np.all(new_X.columns == model.feature_names_in_[model.to_keep_])


def test_select_non_dominated_rejects_invalid_threshold():
    model = SelectNonDominated(threshold=1.1)

    with pytest.raises(ValueError, match="between -1 and 1"):
        model.fit(np.ones((5, 2)))


def test_select_non_dominated_keeps_small_universe():
    X = np.arange(15, dtype=float).reshape(5, 3)
    model = SelectNonDominated(min_n_assets=3)

    model.fit(X)

    np.testing.assert_array_equal(model.get_support(), [True, True, True])


def test_select_non_dominated_considers_negatively_correlated_pair():
    z = np.linspace(-1.0, 1.0, 9)
    X = np.column_stack([0.01 + 0.01 * z, -0.02 * z])

    with_pairs = SelectNonDominated(threshold=0.0).fit(X)
    without_pairs = SelectNonDominated(threshold=-1.0).fit(X)

    np.testing.assert_array_equal(with_pairs.get_support(), [True, True])
    np.testing.assert_array_equal(without_pairs.get_support(), [True, False])


@pytest.mark.parametrize(
    ("min_n_assets", "expected"),
    [
        (None, [True, False, False]),
        (1, [True, False, False]),
        (np.int64(1), [True, False, False]),
        (2, [True, True, False]),
    ],
)
def test_select_non_dominated_honors_exact_minimum(min_n_assets, expected):
    """Stop before adding another front when the minimum is exactly met."""
    z = np.linspace(-1.0, 1.0, 9)
    z = (z - z.mean()) / z.std(ddof=1)
    means = np.array([0.03, 0.015, -0.01])
    variances = np.array([1e-6, 2.5e-5, 1e-4])
    # Successive lower means and higher variances place each asset on a later
    # front; threshold=-1 excludes pair portfolios.
    X = means + z[:, None] * np.sqrt(variances)

    np.testing.assert_allclose(X.mean(axis=0), means)
    np.testing.assert_allclose(X.var(axis=0, ddof=1), variances)
    np.testing.assert_allclose(np.corrcoef(X.T), np.ones((3, 3)))

    model = SelectNonDominated(min_n_assets=min_n_assets, threshold=-1.0).fit(X)

    np.testing.assert_array_equal(model.get_support(), expected)


@pytest.mark.parametrize("min_n_assets", [0, -1, True, np.bool_(True), 1.5, "1"])
def test_select_non_dominated_rejects_invalid_minimum(min_n_assets):
    """Reject non-positive and non-integer asset minimums."""
    X = np.arange(15, dtype=float).reshape(5, 3)

    with pytest.raises(ValueError, match="min_n_assets must be a positive integer"):
        SelectNonDominated(min_n_assets=min_n_assets, threshold=-1.0).fit(X)


def test_select_non_dominated_invalid_refit_preserves_state():
    """Preserve fitted state when a changed minimum is invalid."""
    model = SelectNonDominated(min_n_assets=1, threshold=-1.0).fit(
        np.arange(15, dtype=float).reshape(5, 3)
    )
    n_features_in = model.n_features_in_
    to_keep = model.to_keep_.copy()

    model.set_params(min_n_assets=0)
    with pytest.raises(ValueError, match="min_n_assets must be a positive integer"):
        model.fit(np.arange(20, dtype=float).reshape(5, 4))

    assert model.n_features_in_ == n_features_in
    np.testing.assert_array_equal(model.to_keep_, to_keep)


def test_select_non_dominated_keeps_complete_crossing_front():
    """Keep a complete front when it crosses the requested minimum."""
    z = np.linspace(-1.0, 1.0, 9)
    z = (z - z.mean()) / z.std(ddof=1)
    means = np.array([0.03, 0.02, -0.01])
    variances = np.array([4e-4, 1e-6, 9e-4])
    # Assets 0 and 1 trade return for risk and share the first front; asset 2
    # is dominated by both.
    X = means + z[:, None] * np.sqrt(variances)

    model = SelectNonDominated(min_n_assets=1, threshold=-1.0).fit(X)

    np.testing.assert_array_equal(model.get_support(), [True, True, False])
