import numpy as np
import pandas as pd

from skfolio.model_selection import WalkForward


def assert_split_equal(split, res):
    for i, (train, test) in enumerate(split):
        assert np.array_equal(train, res[i][0])
        assert np.array_equal(test, res[i][1])


def test_walk_forward_with_period(X):
    cv = WalkForward(test_size=2, train_size=3, period="WOM-3FRI")

    list(cv.split(X))

    cv = WalkForward(
        test_size=1, train_size=pd.tseries.offsets.Week(3), period="WOM-3FRI"
    )

    list(cv.split(X))


def test_walk_forward_without_period():
    X = np.random.randn(12, 2)

    cv = WalkForward(
        test_size=4, train_size=1, purged_size=1, reduce_test=True, expend_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0]), np.array([2, 3, 4, 5])),
            (np.array([0, 1, 2, 3, 4]), np.array([6, 7, 8, 9])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), np.array([10, 11])),
        ],
    )
    assert cv.get_n_splits(X) == 3

    cv = WalkForward(
        test_size=4, train_size=1, purged_size=1, reduce_test=False, expend_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0]), np.array([2, 3, 4, 5])),
            (np.array([0, 1, 2, 3, 4]), np.array([6, 7, 8, 9])),
        ],
    )
    assert cv.get_n_splits(X) == 2

    cv = WalkForward(
        test_size=4, train_size=1, purged_size=1, reduce_test=True, expend_train=False
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0]), np.array([2, 3, 4, 5])),
            (np.array([4]), np.array([6, 7, 8, 9])),
            (np.array([8]), np.array([10, 11])),
        ],
    )
    assert cv.get_n_splits(X) == 3

    cv = WalkForward(
        test_size=4, train_size=1, purged_size=1, reduce_test=False, expend_train=False
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0]), np.array([2, 3, 4, 5])),
            (np.array([4]), np.array([6, 7, 8, 9])),
        ],
    )
    assert cv.get_n_splits(X) == 2

    cv = WalkForward(
        test_size=4, train_size=2, purged_size=1, reduce_test=True, expend_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0, 1]), np.array([3, 4, 5, 6])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([7, 8, 9, 10])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([11])),
        ],
    )
    assert cv.get_n_splits(X) == 3

    cv = WalkForward(
        test_size=4, train_size=2, purged_size=0, reduce_test=True, expend_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0, 1]), np.array([2, 3, 4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),
        ],
    )
    assert cv.get_n_splits(X) == 3

    cv = WalkForward(
        test_size=6, train_size=3, purged_size=0, reduce_test=True, expend_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0, 1, 2]), np.array([3, 4, 5, 6, 7, 8])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), np.array([9, 10, 11])),
        ],
    )
    assert cv.get_n_splits(X) == 2
