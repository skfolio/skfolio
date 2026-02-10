"""Test Combinatorial module."""

import math

import numpy as np
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
from skfolio.optimization import InverseVolatility
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


def optimal_folds_number_full_search(
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


def test_optimal_folds_number_weight():
    n_observations = 5000
    target_train_size = 250
    target_n_test_paths = 50

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
    )
    avg_train_size = n_observations / n_folds * (n_folds - n_test_folds)
    n_test_paths = math.comb(n_folds, n_test_folds) * n_test_folds // n_folds

    assert n_folds == 51
    assert n_test_folds == 50
    assert int(avg_train_size) == 98
    assert n_test_paths == 50

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
        weight_train_size=2,
    )
    avg_train_size = n_observations / n_folds * (n_folds - n_test_folds)
    n_test_paths = math.comb(n_folds, n_test_folds) * n_test_folds // n_folds

    assert n_folds == 20
    assert n_test_folds == 19
    assert int(avg_train_size) == 250
    assert n_test_paths == 19


def test_cross_val_predict_and_grid_search(X):
    cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=2)

    model = Pipeline(
        [("pre_selection", SelectKExtremes(k=10)), ("allocation", InverseVolatility())]
    )

    pred = cross_val_predict(model, X, cv=cv)
    assert isinstance(pred, Population)
    assert len(pred) == cv.n_test_paths
