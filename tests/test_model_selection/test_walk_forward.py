import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skfolio.model_selection import WalkForward


def assert_split_equal(split, res):
    for i, (train, test) in enumerate(split):
        assert np.array_equal(train, res[i][0])
        assert np.array_equal(test, res[i][1])


def assert_split_equal_dates(index, split, res):
    for i, (train, test) in enumerate(split):
        assert index[train[0]].date() == res[i][0][0]
        assert index[train[-1]].date() == res[i][0][1]
        assert index[test[0]].date() == res[i][1][0]
        assert index[test[-1]].date() == res[i][1][1]


def _generate(split, index):
    res = []
    for _, (train, test) in enumerate(split):
        res.append(
            (
                (index[train[0]].date(), index[train[-1]].date()),
                (index[test[0]].date(), index[test[-1]].date()),
            )
        )
    return res


@pytest.mark.parametrize(
    "test_size,train_size,freq,freq_offset,previous,reduce_test,expend_train,purged_size,expected",
    [
        (
            2,
            3,
            "WOM-3FRI",
            None,
            True,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 21), dt.date(2022, 4, 13)),
                    (dt.date(2022, 4, 14), dt.date(2022, 6, 16)),
                ),
                (
                    (dt.date(2022, 3, 18), dt.date(2022, 6, 16)),
                    (dt.date(2022, 6, 17), dt.date(2022, 8, 18)),
                ),
                (
                    (dt.date(2022, 5, 20), dt.date(2022, 8, 18)),
                    (dt.date(2022, 8, 19), dt.date(2022, 10, 20)),
                ),
                (
                    (dt.date(2022, 7, 15), dt.date(2022, 10, 20)),
                    (dt.date(2022, 10, 21), dt.date(2022, 12, 15)),
                ),
            ],
        ),
        (
            2,
            3,
            "WOM-3FRI",
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 21), dt.date(2022, 4, 14)),
                    (dt.date(2022, 4, 18), dt.date(2022, 6, 16)),
                ),
                (
                    (dt.date(2022, 3, 18), dt.date(2022, 6, 16)),
                    (dt.date(2022, 6, 17), dt.date(2022, 8, 18)),
                ),
                (
                    (dt.date(2022, 5, 20), dt.date(2022, 8, 18)),
                    (dt.date(2022, 8, 19), dt.date(2022, 10, 20)),
                ),
                (
                    (dt.date(2022, 7, 15), dt.date(2022, 10, 20)),
                    (dt.date(2022, 10, 21), dt.date(2022, 12, 15)),
                ),
            ],
        ),
        (
            4,
            pd.offsets.Week(3),
            "WOM-3FRI",
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 28), dt.date(2022, 2, 17)),
                    (dt.date(2022, 2, 18), dt.date(2022, 6, 16)),
                ),
                (
                    (dt.date(2022, 5, 27), dt.date(2022, 6, 16)),
                    (dt.date(2022, 6, 17), dt.date(2022, 10, 20)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            None,
            True,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 2, 1), dt.date(2022, 7, 29)),
                    (dt.date(2022, 8, 1), dt.date(2022, 9, 29)),
                ),
                (
                    (dt.date(2022, 4, 1), dt.date(2022, 9, 29)),
                    (dt.date(2022, 9, 30), dt.date(2022, 11, 30)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 2, 1), dt.date(2022, 7, 29)),
                    (dt.date(2022, 8, 1), dt.date(2022, 9, 30)),
                ),
                (
                    (dt.date(2022, 4, 1), dt.date(2022, 9, 30)),
                    (dt.date(2022, 10, 3), dt.date(2022, 11, 30)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            None,
            False,
            False,
            False,
            1,
            [
                (
                    (dt.date(2022, 2, 1), dt.date(2022, 7, 28)),
                    (dt.date(2022, 8, 1), dt.date(2022, 9, 30)),
                ),
                (
                    (dt.date(2022, 4, 1), dt.date(2022, 9, 29)),
                    (dt.date(2022, 10, 3), dt.date(2022, 11, 30)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            None,
            False,
            True,
            False,
            0,
            [
                (
                    (dt.date(2022, 2, 1), dt.date(2022, 7, 29)),
                    (dt.date(2022, 8, 1), dt.date(2022, 9, 30)),
                ),
                (
                    (dt.date(2022, 4, 1), dt.date(2022, 9, 30)),
                    (dt.date(2022, 10, 3), dt.date(2022, 11, 30)),
                ),
                (
                    (dt.date(2022, 6, 1), dt.date(2022, 11, 30)),
                    (dt.date(2022, 12, 1), dt.date(2022, 12, 28)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            None,
            False,
            False,
            True,
            0,
            [
                (
                    (dt.date(2022, 1, 3), dt.date(2022, 7, 29)),
                    (dt.date(2022, 8, 1), dt.date(2022, 9, 30)),
                ),
                (
                    (dt.date(2022, 1, 3), dt.date(2022, 9, 30)),
                    (dt.date(2022, 10, 3), dt.date(2022, 11, 30)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            dt.timedelta(days=2),
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 3), dt.date(2022, 7, 1)),
                    (dt.date(2022, 7, 5), dt.date(2022, 9, 2)),
                ),
                (
                    (dt.date(2022, 3, 3), dt.date(2022, 9, 2)),
                    (dt.date(2022, 9, 6), dt.date(2022, 11, 2)),
                ),
            ],
        ),
        (
            2,
            6,
            "MS",
            pd.offsets.BDay(2),
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 4), dt.date(2022, 7, 1)),
                    (dt.date(2022, 7, 5), dt.date(2022, 9, 2)),
                ),
                (
                    (dt.date(2022, 3, 3), dt.date(2022, 9, 2)),
                    (dt.date(2022, 9, 6), dt.date(2022, 11, 2)),
                ),
            ],
        ),
        (
            1,
            48,
            pd.offsets.Week(weekday=4),
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 7), dt.date(2022, 12, 8)),
                    (dt.date(2022, 12, 9), dt.date(2022, 12, 15)),
                ),
                (
                    (dt.date(2022, 1, 14), dt.date(2022, 12, 15)),
                    (dt.date(2022, 12, 16), dt.date(2022, 12, 22)),
                ),
            ],
        ),
        (
            1,
            pd.offsets.Week(48),
            pd.offsets.Week(weekday=4),
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 1, 7), dt.date(2022, 12, 8)),
                    (dt.date(2022, 12, 9), dt.date(2022, 12, 15)),
                ),
                (
                    (dt.date(2022, 1, 14), dt.date(2022, 12, 15)),
                    (dt.date(2022, 12, 16), dt.date(2022, 12, 22)),
                ),
            ],
        ),
        (
            1,
            pd.DateOffset(months=1),
            pd.offsets.QuarterEnd(),
            None,
            False,
            False,
            False,
            0,
            [
                (
                    (dt.date(2022, 2, 28), dt.date(2022, 3, 30)),
                    (dt.date(2022, 3, 31), dt.date(2022, 6, 29)),
                ),
                (
                    (dt.date(2022, 5, 27), dt.date(2022, 6, 29)),
                    (dt.date(2022, 6, 30), dt.date(2022, 9, 29)),
                ),
            ],
        ),
        (
            1,
            pd.DateOffset(years=1),
            pd.offsets.QuarterEnd(),
            None,
            False,
            False,
            False,
            0,
            [],
        ),
    ],
)
def test_walk_forward_with_period(
    X_small,
    test_size,
    train_size,
    freq,
    freq_offset,
    previous,
    reduce_test,
    expend_train,
    purged_size,
    expected,
):
    cv = WalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        reduce_test=reduce_test,
        expend_train=expend_train,
        purged_size=purged_size,
    )
    assert_split_equal_dates(X_small.index, cv.split(X_small), expected)


@pytest.mark.parametrize(
    "test_size,train_size,freq,previous,expected",
    [
        (
            1,
            4,
            "QS",
            True,
            [
                (
                    (dt.date(2020, 4, 1), dt.date(2021, 3, 31)),
                    (dt.date(2021, 4, 1), dt.date(2021, 6, 30)),
                ),
                (
                    (dt.date(2020, 7, 1), dt.date(2021, 6, 30)),
                    (dt.date(2021, 7, 1), dt.date(2021, 9, 30)),
                ),
                (
                    (dt.date(2020, 10, 1), dt.date(2021, 9, 30)),
                    (dt.date(2021, 10, 1), dt.date(2021, 12, 30)),
                ),
                (
                    (dt.date(2020, 12, 31), dt.date(2021, 12, 30)),
                    (dt.date(2021, 12, 31), dt.date(2022, 3, 31)),
                ),
                (
                    (dt.date(2021, 4, 1), dt.date(2022, 3, 31)),
                    (dt.date(2022, 4, 1), dt.date(2022, 6, 30)),
                ),
                (
                    (dt.date(2021, 7, 1), dt.date(2022, 6, 30)),
                    (dt.date(2022, 7, 1), dt.date(2022, 9, 29)),
                ),
            ],
        ),
        (
            1,
            pd.DateOffset(years=1),
            "QS",
            False,
            [
                (
                    (dt.date(2020, 4, 1), dt.date(2021, 3, 31)),
                    (dt.date(2021, 4, 1), dt.date(2021, 6, 30)),
                ),
                (
                    (dt.date(2020, 7, 1), dt.date(2021, 6, 30)),
                    (dt.date(2021, 7, 1), dt.date(2021, 9, 30)),
                ),
                (
                    (dt.date(2020, 10, 1), dt.date(2021, 9, 30)),
                    (dt.date(2021, 10, 1), dt.date(2021, 12, 31)),
                ),
                (
                    (dt.date(2020, 12, 31), dt.date(2021, 12, 31)),
                    (dt.date(2022, 1, 3), dt.date(2022, 3, 31)),
                ),
                (
                    (dt.date(2021, 4, 1), dt.date(2022, 3, 31)),
                    (dt.date(2022, 4, 1), dt.date(2022, 6, 30)),
                ),
                (
                    (dt.date(2021, 7, 1), dt.date(2022, 6, 30)),
                    (dt.date(2022, 7, 1), dt.date(2022, 9, 30)),
                ),
            ],
        ),
    ],
)
def test_walk_forward_with_period_long(
    X_medium, test_size, train_size, freq, previous, expected
):
    cv = WalkForward(
        test_size=test_size, train_size=train_size, freq=freq, previous=previous
    )
    assert_split_equal_dates(X_medium.index, cv.split(X_medium), expected)


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
