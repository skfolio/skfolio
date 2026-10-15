"""Test Combinatorial module."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from skfolio import Population
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    cross_val_predict,
    optimal_folds_number,
)
from skfolio.model_selection._combinatorial import (
    _MAX_COMBINATIONS,
    _avg_train_size,
    _n_test_paths,
)
from skfolio.optimization import EqualWeighted, InverseVolatility
from skfolio.pre_selection import SelectKExtremes


def assert_split_equal(split, res):
    for i, (train, tests) in enumerate(split):
        assert np.array_equal(train, res[i][0])
        for j, test in enumerate(tests):
            assert np.array_equal(test, res[i][1][j])


def test_combinatorial_purged_cv():
    X = np.random.randn(12, 2)

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=0)

    assert_split_equal(
        cv.split(X),
        [
            (
                np.array([8, 9, 10, 11]),
                [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])],
            ),
            (
                np.array([4, 5, 6, 7]),
                [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3]),
                [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )
    assert cv.n_splits == 3
    assert cv.n_test_paths == 2
    assert np.array_equal(cv.test_set_index, np.array([[0, 1], [0, 2], [1, 2]]))
    assert np.array_equal(
        cv.binary_train_test_sets,
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    assert np.array_equal(cv.recombined_paths, np.array([[0, 1], [0, 2], [1, 2]]))
    assert np.array_equal(cv.get_path_ids(), np.array([[0, 0], [1, 0], [1, 1]]))

    assert cv.plot_train_test_folds()
    assert cv.plot_train_test_index(X)
    cv.summary(X)

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=0)
    assert_split_equal(
        cv.split(X),
        [
            (np.array([9, 10, 11]), [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]),
            (np.array([5, 6]), [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])]),
            (np.array([0, 1, 2]), [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])]),
        ],
    )

    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=1)
    assert_split_equal(
        cv.split(X),
        [
            (np.array([9, 10, 11]), [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]),
            (np.array([5, 6, 7]), [np.array([0, 1, 2, 3]), np.array([8, 9, 10, 11])]),
            (
                np.array([0, 1, 2, 3]),
                [np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )

    cv = CombinatorialPurgedCV(n_folds=5, n_test_folds=2, purged_size=0, embargo_size=0)
    assert_split_equal(
        cv.split(X),
        [
            (
                np.array([4, 5, 6, 7, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([2, 3])],
            ),
            (
                np.array([2, 3, 6, 7, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([4, 5])],
            ),
            (
                np.array([2, 3, 4, 5, 8, 9, 10, 11]),
                [np.array([0, 1]), np.array([6, 7])],
            ),
            (
                np.array([2, 3, 4, 5, 6, 7]),
                [np.array([0, 1]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 6, 7, 8, 9, 10, 11]),
                [np.array([2, 3]), np.array([4, 5])],
            ),
            (
                np.array([0, 1, 4, 5, 8, 9, 10, 11]),
                [np.array([2, 3]), np.array([6, 7])],
            ),
            (
                np.array([0, 1, 4, 5, 6, 7]),
                [np.array([2, 3]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3, 8, 9, 10, 11]),
                [np.array([4, 5]), np.array([6, 7])],
            ),
            (
                np.array([0, 1, 2, 3, 6, 7]),
                [np.array([4, 5]), np.array([8, 9, 10, 11])],
            ),
            (
                np.array([0, 1, 2, 3, 4, 5]),
                [np.array([6, 7]), np.array([8, 9, 10, 11])],
            ),
        ],
    )
    assert cv.n_splits == 10
    assert cv.n_test_paths == 4
    assert np.array_equal(
        cv.test_set_index,
        np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
            ]
        ),
    )
    assert np.array_equal(
        cv.binary_train_test_sets,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            ]
        ),
    )
    assert np.array_equal(
        cv.recombined_paths,
        np.array(
            [[0, 1, 2, 3], [0, 4, 5, 6], [1, 4, 7, 8], [2, 5, 7, 9], [3, 6, 8, 9]]
        ),
    )
    assert np.array_equal(
        cv.get_path_ids(),
        np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [1, 1],
                [2, 1],
                [3, 1],
                [2, 2],
                [3, 2],
                [3, 3],
            ]
        ),
    )


class TestCombinatorialPurgedCVMaxCombinations:
    """Tests for the _MAX_COMBINATIONS guard in CombinatorialPurgedCV."""

    def test_exceeds_max_combinations(self):
        """n_folds=20, n_test_folds=10 produces C(20,10)=184,756 splits which
        exceeds _MAX_COMBINATIONS and should raise."""
        with pytest.raises(ValueError, match="exceeds the maximum allowed"):
            CombinatorialPurgedCV(n_folds=20, n_test_folds=10)

    def test_error_message_contains_split_count(self):
        n_folds, n_test_folds = 20, 10
        n_combinations = math.comb(n_folds, n_test_folds)
        with pytest.raises(ValueError, match=f"{n_combinations:,}"):
            CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)

    def test_error_message_contains_max(self):
        with pytest.raises(ValueError, match=f"{_MAX_COMBINATIONS:,}"):
            CombinatorialPurgedCV(n_folds=20, n_test_folds=10)

    def test_error_message_mentions_misconfiguration(self):
        with pytest.raises(ValueError, match="misconfiguration"):
            CombinatorialPurgedCV(n_folds=20, n_test_folds=10)

    def test_below_max_combinations_ok(self):
        """n_folds=15, n_test_folds=2 produces C(15,2)=105 splits, well within
        the limit."""
        cv = CombinatorialPurgedCV(n_folds=15, n_test_folds=2)
        assert cv.n_splits == math.comb(15, 2)

    def test_just_below_max_combinations_ok(self):
        """Find a combination just under the limit and verify it is accepted."""
        # C(18, 9) = 48,620 — well within 100,000
        cv = CombinatorialPurgedCV(n_folds=18, n_test_folds=9)
        assert cv.n_splits == math.comb(18, 9)

    def test_symmetric_both_sides(self):
        """C(n, k) == C(n, n-k), so n_test_folds near 1 should be fine even
        for large n_folds, while n_test_folds near n_folds/2 blows up."""
        # C(50, 2) = 1,225 — fine
        cv = CombinatorialPurgedCV(n_folds=50, n_test_folds=2)
        assert cv.n_splits == 1225

        # C(50, 25) is astronomically large — should raise
        with pytest.raises(ValueError, match="exceeds the maximum allowed"):
            CombinatorialPurgedCV(n_folds=50, n_test_folds=25)


def _optimal_folds_number_full_search(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
) -> tuple[int, int]:
    def _cost(
        x: int,
        y: int,
    ) -> float:
        n_test_paths = _n_test_paths(n_folds=x, n_test_folds=y)
        avg_train_size = _avg_train_size(
            n_observations=n_observations, n_folds=x, n_test_folds=y
        )
        return (
            abs(n_test_paths - target_n_test_paths) / target_n_test_paths
            + abs(avg_train_size - target_train_size) / target_train_size
        )

    res = []
    costs = []
    for n_folds in range(3, n_observations + 1):
        for n_test_folds in range(2, n_folds):
            res.append((n_folds, n_test_folds))
            costs.append(_cost(x=n_folds, y=n_test_folds))
    i = np.argmin(costs)
    return res[i]


@pytest.mark.parametrize(
    "n_observations,target_n_test_paths,target_train_size,expected",
    [
        (10, 10, 1, (10, 9)),
        (10, 2, 100, (3, 2)),
        (10, 2, 5, (3, 2)),
        (100, 20, 10, (21, 20)),
        (100, 5, 30, (6, 5)),
        (1000, 300, 50, (26, 24)),
    ],
)
def test_optimal_folds_number(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
    expected: tuple[int, int],
):
    res = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
    )
    assert res == expected
    if n_observations <= 100:
        assert res == _optimal_folds_number_full_search(
            n_observations=n_observations,
            target_train_size=target_train_size,
            target_n_test_paths=target_n_test_paths,
        )


def test_optimal_folds_number_weight():
    n_observations = 500
    target_train_size = 50
    target_n_test_paths = 20

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
    )
    avg_train_size = n_observations / n_folds * (n_folds - n_test_folds)
    n_test_paths = math.comb(n_folds, n_test_folds) * n_test_folds // n_folds

    assert n_folds == 21
    assert n_test_folds == 20
    assert int(avg_train_size) == 23
    assert n_test_paths == 20

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
        weight_train_size=2,
    )
    avg_train_size = n_observations / n_folds * (n_folds - n_test_folds)
    n_test_paths = math.comb(n_folds, n_test_folds) * n_test_folds // n_folds

    assert n_folds == 10
    assert n_test_folds == 9
    assert int(avg_train_size) == 50
    assert n_test_paths == 9


def test_cross_val_predict_and_grid_search(X):
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=2)

    model = Pipeline(
        [("pre_selection", SelectKExtremes(k=10)), ("allocation", InverseVolatility())]
    )

    pred = cross_val_predict(model, X, cv=cv)
    assert isinstance(pred, Population)
    assert len(pred) == cv.n_test_paths


def test_combinatorial_purged_cv_split_returns_lists():
    """Test that split() yields lists of test indices for multi-path backtesting."""
    X = np.random.randn(12, 2)
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=0)

    splits = list(cv.split(X))

    # Should have 3 splits
    assert len(splits) == 3

    for train, test in splits:
        # test should be a list for multi-path backtesting
        assert isinstance(test, list)
        assert len(test) == 2  # 2 test folds

        # Each test element should be an array
        for test_array in test:
            assert isinstance(test_array, np.ndarray)

        # train should be an array
        assert isinstance(train, np.ndarray)

        # test arrays and train should be non-overlapping
        test_concat = np.concatenate(test)
        assert len(np.intersect1d(train, test_concat)) == 0


def test_combinatorial_purged_cv_get_n_splits():
    """Test that get_n_splits returns correct number of splits."""
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=0, embargo_size=0)

    assert cv.get_n_splits() == cv.n_splits
    assert cv.get_n_splits() == 3
    # scikit-learn compatible signature: X, y, groups are accepted but ignored
    assert cv.get_n_splits(X=None, y=None, groups=None) == 3


def test_cross_val_predict_concatenated_indices(X):
    """Test that cross_val_predict correctly handles multi-path test indices."""
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=2)

    model = Pipeline(
        [("pre_selection", SelectKExtremes(k=10)), ("allocation", InverseVolatility())]
    )

    # cross_val_predict should handle list test indices gracefully
    pred = cross_val_predict(model, X, cv=cv)

    # Result should be a Population with correct number of paths
    assert isinstance(pred, Population)
    assert len(pred) == cv.n_test_paths

    # Each path should be a MultiPeriodPortfolio
    for portfolio in pred:
        assert hasattr(portfolio, "name")
        # Each portfolio should have correct number of folds
        assert len(portfolio.portfolios) == cv.n_folds


def test_combinatorial_purged_cv_regression():
    """Regression test: ensure CombinatorialPurgedCV returns lists for multi-path."""
    X = np.random.randn(20, 5)
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2, purged_size=0, embargo_size=0)

    for _, test in cv.split(X):
        # test should be a list for multi-path backtesting
        assert isinstance(test, list), (
            "split() should yield lists for multi-path backtesting"
        )
        assert len(test) == cv.n_test_folds
        # Should contain valid indices
        for test_array in test:
            assert np.all((test_array >= 0) & (test_array < len(X)))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_folds": 3.5}, "The number of folds must be of Integral type"),
        ({"n_folds": 2, "n_test_folds": 2}, "`n_folds` must be at least 3"),
        ({"n_folds": 3, "n_test_folds": 1}, "`n_test_folds` must at least 2"),
        (
            {"n_folds": 3, "n_test_folds": 3},
            "requires `n_folds` to be greater than `n_test_folds`",
        ),
        ({"n_folds": 3, "n_test_folds": 2, "purged_size": -1}, "`purged_size`"),
        ({"n_folds": 3, "n_test_folds": 2, "embargo_size": -1}, "`embargo_size`"),
    ],
)
def test_combinatorial_purged_cv_invalid_init(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CombinatorialPurgedCV(**kwargs)


def test_combinatorial_purged_cv_split_rejects_too_large_purge_and_embargo():
    X = np.zeros((12, 2))
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=2, embargo_size=1)
    with pytest.raises(
        ValueError,
        match="sum of `purged_size` and `embargo_size` must be smaller than the size",
    ):
        list(cv.split(X))


@pytest.mark.parametrize(
    "n_samples,n_folds,n_test_folds,expected_fold_sizes",
    [
        (11, 5, 2, [2, 2, 2, 2, 3]),
        (12, 5, 2, [2, 2, 2, 2, 4]),
        (13, 5, 2, [2, 2, 2, 2, 5]),
        (89, 10, 2, [8] * 9 + [17]),
        (14, 5, 3, [2, 2, 2, 2, 6]),
        (12, 4, 2, [3, 3, 3, 3]),
        (23, 6, 2, [3, 3, 3, 3, 3, 8]),
    ],
)
def test_combinatorial_purged_cv_fold_boundaries_and_paths(
    n_samples, n_folds, n_test_folds, expected_fold_sizes
):
    """Keep the remainder in the last fold and cover each observation once per path."""
    X = pd.DataFrame(
        np.zeros((n_samples, 2)),
        index=pd.date_range("2020-01-01", periods=n_samples),
    )
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)
    indices = np.arange(n_samples)
    expected_folds = np.split(indices, np.cumsum(expected_fold_sizes)[:-1])
    for (train, tests), fold_ids in zip(
        cv.split(X), itertools.combinations(range(n_folds), n_test_folds), strict=True
    ):
        for test, fold_id in zip(tests, fold_ids, strict=True):
            np.testing.assert_array_equal(test, expected_folds[fold_id])
        expected_test = np.concatenate([expected_folds[i] for i in fold_ids])
        np.testing.assert_array_equal(train, np.setdiff1d(indices, expected_test))

    pred = cross_val_predict(EqualWeighted(), X, cv=cv)
    assert len(pred) == math.comb(n_folds - 1, n_test_folds - 1)
    for path in pred:
        np.testing.assert_array_equal(path.observations, X.index)
        assert [fold.n_observations for fold in path.portfolios] == expected_fold_sizes


@pytest.mark.parametrize(
    "purged_size,embargo_size,test_fold_ids,expected_train",
    [
        (0, 0, (0, 5), np.arange(4, 20)),
        (1, 0, (0, 5), np.arange(5, 19)),
        (0, 1, (0, 5), np.arange(5, 20)),
        (1, 1, (0, 5), np.arange(6, 19)),
        (1, 1, (4, 5), np.arange(15)),
        (1, 1, (0, 1), np.arange(10, 29)),
    ],
)
def test_combinatorial_purged_cv_uneven_folds_purge_and_embargo(
    purged_size, embargo_size, test_fold_ids, expected_train
):
    """Purge around test blocks, including an enlarged final fold and adjacent folds."""
    cv = CombinatorialPurgedCV(
        n_folds=6,
        n_test_folds=2,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )
    splits = list(cv.split(np.zeros((29, 2))))
    split_id = list(itertools.combinations(range(6), 2)).index(test_fold_ids)
    train, tests = splits[split_id]
    expected_folds = np.split(np.arange(29), [4, 8, 12, 16, 20])

    np.testing.assert_array_equal(train, expected_train)
    for test, fold_id in zip(tests, test_fold_ids, strict=True):
        np.testing.assert_array_equal(test, expected_folds[fold_id])
    # Adjacent test folds form one block: their shared boundary must not be purged.
    assert np.all(cv.index_train_test_[np.concatenate(tests), split_id] == 1)
