from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skfolio import MultiPeriodPortfolio
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.model_selection import _walk_forward as walk_forward_module
from skfolio.optimization import InverseVolatility
from skfolio.pre_selection import SelectKExtremes


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
    "test_size,train_size,freq,freq_offset,previous,reduce_test,expand_train,purged_size,expected",
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
    expand_train,
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
        expand_train=expand_train,
        purged_size=purged_size,
    )
    assert_split_equal_dates(X_small.index, cv.split(X_small), expected)
    assert cv.get_n_splits(X_small) == len(list(cv.split(X_small)))


def test_walk_forward_with_train_offset_purges_training():
    """Remove purged observations from offset-based training windows only."""
    X = pd.DataFrame(
        np.arange(240).reshape(120, 2),
        index=pd.date_range("2026-01-01", periods=120, freq="D"),
    )
    # We cover both offset types and every alignment, window, and final-fold policy.
    for train_size, previous, expand_train, reduce_test in itertools.product(
        (pd.DateOffset(months=1), dt.timedelta(days=31)),
        (False, True),
        (False, True),
        (False, True),
    ):
        unpurged_cv = WalkForward(
            test_size=1,
            train_size=train_size,
            freq="MS",
            previous=previous,
            reduce_test=reduce_test,
            expand_train=expand_train,
            purged_size=0,
        )
        purged_cv = WalkForward(
            test_size=1,
            train_size=train_size,
            freq="MS",
            previous=previous,
            reduce_test=reduce_test,
            expand_train=expand_train,
            purged_size=1,
        )
        unpurged_splits = list(unpurged_cv.split(X))
        purged_splits = list(purged_cv.split(X))

        assert purged_cv.get_n_splits(X) == len(purged_splits)
        assert len(purged_splits) == len(unpurged_splits)
        for (unpurged_train, unpurged_test), (
            purged_train,
            purged_test,
        ) in zip(unpurged_splits, purged_splits, strict=True):
            np.testing.assert_array_equal(purged_train, unpurged_train[:-1])
            np.testing.assert_array_equal(purged_test, unpurged_test)

        # We check explicit expected indices so errors shared by both splitters
        # cannot pass.
        first_train, first_test = purged_splits[0]
        np.testing.assert_array_equal(first_train, np.arange(30))
        np.testing.assert_array_equal(first_test, np.arange(31, 59))


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
    assert cv.get_n_splits(X_medium) == len(list(cv.split(X_medium)))


@pytest.mark.parametrize(
    "test_size,train_size,freq,purged_size,match",
    [
        pytest.param(
            0,
            2,
            None,
            0,
            r"test_size must be a positive integer",
            id="test-size-zero",
        ),
        pytest.param(
            -1,
            2,
            None,
            0,
            r"test_size must be a positive integer",
            id="test-size-negative",
        ),
        pytest.param(
            1.5,
            2,
            None,
            0,
            r"test_size must be a positive integer",
            id="test-size-float",
        ),
        pytest.param(
            True,
            2,
            None,
            0,
            r"test_size must be a positive integer",
            id="test-size-bool",
        ),
        pytest.param(
            np.bool_(True),
            2,
            None,
            0,
            r"test_size must be a positive integer",
            id="test-size-numpy-bool",
        ),
        pytest.param(
            2,
            0,
            None,
            0,
            r"train_size must be a positive integer",
            id="train-size-zero",
        ),
        pytest.param(
            2,
            -1,
            None,
            0,
            r"train_size must be a positive integer",
            id="train-size-negative",
        ),
        pytest.param(
            2,
            1.5,
            None,
            0,
            r"train_size must be an integer when freq is None",
            id="train-size-float",
        ),
        pytest.param(
            2,
            True,
            None,
            0,
            r"train_size must be an integer when freq is None",
            id="train-size-bool",
        ),
        pytest.param(
            2,
            np.bool_(True),
            None,
            0,
            r"train_size must be an integer when freq is None",
            id="train-size-numpy-bool",
        ),
        pytest.param(
            2,
            "6M",
            "D",
            0,
            r"train_size must be an integer, pandas DateOffset",
            id="train-size-string-with-frequency",
        ),
        pytest.param(
            2,
            True,
            "D",
            0,
            r"train_size must be an integer, pandas DateOffset",
            id="train-size-bool-with-frequency",
        ),
        pytest.param(
            2,
            2,
            None,
            -1,
            r"purged_size must be a non-negative integer",
            id="purged-size-negative",
        ),
        pytest.param(
            2,
            2,
            None,
            1.5,
            r"purged_size must be a non-negative integer",
            id="purged-size-float",
        ),
        pytest.param(
            2,
            2,
            None,
            True,
            r"purged_size must be a non-negative integer",
            id="purged-size-bool",
        ),
        pytest.param(
            2,
            2,
            None,
            np.bool_(True),
            r"purged_size must be a non-negative integer",
            id="purged-size-numpy-bool",
        ),
    ],
)
@pytest.mark.parametrize("method_name", ["split", "get_n_splits"])
def test_walk_forward_rejects_invalid_window_sizes(
    test_size, train_size, freq, purged_size, match, method_name
):
    """Reject invalid window sizes before splitting or counting folds."""
    X = pd.DataFrame(
        np.arange(20).reshape(10, 2),
        index=pd.date_range("2026-01-01", periods=10),
    )
    cv = WalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=freq,
        purged_size=purged_size,
    )

    with pytest.raises(ValueError, match=match):
        if method_name == "split":
            list(cv.split(X))
        else:
            cv.get_n_splits(X)


@pytest.mark.parametrize("freq", [None, "D"])
def test_walk_forward_accepts_numpy_integer_window_sizes(freq):
    """Accept NumPy integer sizes for index- and calendar-based windows."""
    X = pd.DataFrame(
        np.arange(20).reshape(10, 2),
        index=pd.date_range("2026-01-01", periods=10),
    )
    cv = WalkForward(
        test_size=np.int64(2),
        train_size=np.int32(2),
        freq=freq,
        purged_size=np.int64(1),
    )

    splits = list(cv.split(X))
    assert splits
    assert cv.get_n_splits(X) == len(splits)
    assert all(train.size > 0 and test.size > 0 for train, test in splits)


@pytest.mark.parametrize("n_samples", [2, 4, 5])
@pytest.mark.parametrize("reduce_test", [False, True])
def test_walk_forward_rejects_insufficient_observations(n_samples, reduce_test):
    """Reject observation schedules with no possible test observation."""
    # We cover inputs shorter than training, equal to training, and equal to
    # training plus purging.
    X = np.zeros((n_samples, 2))
    cv = WalkForward(
        test_size=2,
        train_size=4,
        purged_size=1,
        reduce_test=reduce_test,
    )
    match = r"must be less than the number of observations"

    with pytest.raises(ValueError, match=match) as count_error:
        cv.get_n_splits(X)
    with pytest.raises(ValueError, match=match) as split_error:
        cv.split(X)
    assert str(count_error.value) == str(split_error.value)


@pytest.mark.parametrize(("reduce_test", "expected_n_splits"), [(False, 0), (True, 1)])
def test_walk_forward_accepts_single_test_observation(reduce_test, expected_n_splits):
    """Accept the smallest schedule that leaves a test observation."""
    X = np.zeros((6, 2))
    cv = WalkForward(
        test_size=2,
        train_size=4,
        purged_size=1,
        reduce_test=reduce_test,
    )

    # The lone test observation forms a fold only when partial tests are retained.
    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == expected_n_splits
    if reduce_test:
        train, test = splits[0]
        np.testing.assert_array_equal(train, np.arange(4))
        np.testing.assert_array_equal(test, np.array([5]))


def test_walk_forward_without_period():
    X = np.random.randn(12, 2)

    cv = WalkForward(
        test_size=4, train_size=1, purged_size=1, reduce_test=True, expand_train=True
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
        test_size=4, train_size=1, purged_size=1, reduce_test=False, expand_train=True
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
        test_size=4, train_size=1, purged_size=1, reduce_test=True, expand_train=False
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
        test_size=4, train_size=1, purged_size=1, reduce_test=False, expand_train=False
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
        test_size=4, train_size=2, purged_size=1, reduce_test=True, expand_train=True
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
        test_size=4, train_size=2, purged_size=0, reduce_test=True, expand_train=True
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
        test_size=6, train_size=3, purged_size=0, reduce_test=True, expand_train=True
    )
    assert_split_equal(
        cv.split(X),
        [
            (np.array([0, 1, 2]), np.array([3, 4, 5, 6, 7, 8])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), np.array([9, 10, 11])),
        ],
    )
    assert cv.get_n_splits(X) == 2


@pytest.mark.parametrize(
    "test_size,train_size,freq,freq_offset,previous,reduce_test,expand_train,purged_size",
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
        ),
    ],
)
def test_cross_val_predict_and_grid_search(
    X_medium,
    test_size,
    train_size,
    freq,
    freq_offset,
    previous,
    reduce_test,
    expand_train,
    purged_size,
):
    X_test = X_medium.iloc[:, :6]
    cv = WalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        reduce_test=reduce_test,
        expand_train=expand_train,
        purged_size=purged_size,
    )

    model = Pipeline(
        [("pre_selection", SelectKExtremes(k=5)), ("allocation", InverseVolatility())]
    )

    pred = cross_val_predict(model, X_test, cv=cv)
    assert isinstance(pred, MultiPeriodPortfolio)
    assert len(pred) == cv.get_n_splits(X_test)

    gs = GridSearchCV(estimator=model, cv=cv, param_grid={"pre_selection__k": [2, 3]})
    gs.fit(X_test)
    assert gs.best_estimator_


def test_expend_train_removed():
    with pytest.raises(TypeError, match="expend_train"):
        WalkForward(test_size=2, train_size=3, expend_train=True)


@pytest.mark.parametrize(
    ("cv", "match"),
    [
        (WalkForward(test_size=1.5, train_size=2), "test_size"),
        (WalkForward(test_size=2, train_size=1.5), "train_size"),
    ],
)
def test_walk_forward_rejects_non_integer_sizes_without_frequency(cv, match):
    with pytest.raises(ValueError, match=match):
        list(cv.split(np.ones((8, 2))))


def test_walk_forward_frequency_requires_datetime_index():
    cv = WalkForward(test_size=1, train_size=2, freq="MS")
    X = np.ones((8, 2))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        list(cv.split(X))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        cv.get_n_splits(X)


def test_walk_forward_get_n_splits_requires_data():
    with pytest.raises(ValueError, match="should not be None"):
        WalkForward(test_size=2, train_size=3).get_n_splits()


def test_walk_forward_offset_with_date_based_training_window(X_medium):
    cv = WalkForward(
        test_size=1,
        train_size=pd.DateOffset(months=3),
        freq="MS",
        freq_offset=pd.offsets.BDay(1),
        reduce_test=True,
        expand_train=True,
    )

    splits = list(cv.split(X_medium))
    boundaries = _generate(splits, X_medium.index)

    assert len(splits) == cv.get_n_splits(X_medium) == 33
    assert boundaries[0] == (
        (dt.date(2020, 1, 2), dt.date(2020, 4, 1)),
        (dt.date(2020, 4, 2), dt.date(2020, 5, 1)),
    )
    assert boundaries[-1] == (
        (dt.date(2020, 1, 2), dt.date(2022, 12, 1)),
        (dt.date(2022, 12, 2), dt.date(2022, 12, 28)),
    )


def test_walk_forward_get_n_splits_offset_train_window_before_index(X_small):
    cv = WalkForward(test_size=1, train_size=pd.DateOffset(years=100), freq="MS")
    assert cv.get_n_splits(X_small) == 0


def test_walk_forward_get_n_splits_offset_train_window_leaves_no_test_window(
    X_small,
):
    # X_small covers 2022 (first observation 2022-01-03): the "MS" schedule has 11
    # rebalancing dates (Feb..Dec). Shifting them back 6 months leaves the first
    # valid training start at position 6, which is not before the last allowed
    # start `n - test_size = 11 - 5 = 6`, so no test window fits.
    cv = WalkForward(test_size=5, train_size=pd.DateOffset(months=6), freq="MS")
    assert cv.get_n_splits(X_small) == 0
    assert (
        WalkForward(
            test_size=4, train_size=pd.DateOffset(months=6), freq="MS"
        ).get_n_splits(X_small)
        == 1
    )


@pytest.mark.parametrize("freq", ["D", "B"])
@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("test_size", [1, 3])
@pytest.mark.parametrize("reduce_test", [False, True])
@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("purged_size", [0, 1])
@pytest.mark.parametrize(
    "train_size", [14, pd.Timedelta(days=14)], ids=["train_periods", "train_offset"]
)
def test_walk_forward_skips_periods_without_observations(
    freq, previous, test_size, reduce_test, expand_train, purged_size, train_size
):
    """Periods without observations (weekends, holidays) give no empty test fold."""
    index = pd.bdate_range("2024-01-01", periods=60)
    # We drop a few weekdays to mimic market holidays.
    index = index.drop(index[[7, 8, 20, 33]])
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(len(index), 2)), index=index)
    cv = WalkForward(
        test_size=test_size,
        train_size=train_size,
        freq=freq,
        previous=previous,
        reduce_test=reduce_test,
        expand_train=expand_train,
        purged_size=purged_size,
    )

    splits = list(cv.split(X))

    assert splits
    assert cv.get_n_splits(X) == len(splits)
    assert all(test.size > 0 for _, test in splits)
    # The test folds tile the observations without gaps or overlaps.
    tests = np.concatenate([test for _, test in splits])
    assert np.array_equal(tests, np.arange(tests[0], tests[0] + len(tests)))
    cross_val_predict(InverseVolatility(), X, cv=cv)


@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("purged_size", [0, 999])
@pytest.mark.parametrize(
    "train_size",
    [1000, pd.Timedelta(minutes=1000)],
    ids=["train_periods", "train_offset"],
)
def test_walk_forward_get_n_splits_does_not_generate_splits(
    monkeypatch, expand_train, purged_size, train_size
):
    """Counting must not call either the public or private split generator."""
    n_samples = 10_000
    X = pd.DataFrame(
        {"x": np.zeros(n_samples)},
        index=pd.date_range("2024-01-01", periods=n_samples, freq="min"),
    )
    cv = WalkForward(
        test_size=1,
        train_size=train_size,
        freq="min",
        expand_train=expand_train,
        purged_size=purged_size,
    )

    def unexpected_split(*args, **kwargs):
        pytest.fail("get_n_splits must not generate train/test splits")

    monkeypatch.setattr(cv, "split", unexpected_split)
    monkeypatch.setattr(walk_forward_module, "_split_by_calendar", unexpected_split)

    assert cv.get_n_splits(X) == n_samples - 1000 - 1


@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("reduce_test", [False, True])
@pytest.mark.parametrize("purged_size", [0, 1])
@pytest.mark.parametrize("train_size", [1, pd.Timedelta(days=30)])
def test_walk_forward_shifted_date_before_data(
    expand_train, reduce_test, purged_size, train_size
):
    """A missing previous observation must not leak the last observation into training."""
    X = pd.DataFrame({"x": np.zeros(90)}, index=pd.date_range("2024-01-01", periods=90))
    cv = WalkForward(
        train_size=train_size,
        test_size=1,
        freq="MS",
        freq_offset=-pd.Timedelta(days=1),
        previous=True,
        expand_train=expand_train,
        reduce_test=reduce_test,
        purged_size=purged_size,
    )

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == (2 if reduce_test else 1)
    np.testing.assert_array_equal(splits[0][0], np.arange(30 - purged_size))
    np.testing.assert_array_equal(splits[0][1], np.arange(30, 59))
    if reduce_test:
        train_start = 0 if expand_train else (30 if isinstance(train_size, int) else 29)
        np.testing.assert_array_equal(
            splits[1][0], np.arange(train_start, 59 - purged_size)
        )
        np.testing.assert_array_equal(splits[1][1], np.arange(59, 90))


@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("reduce_test", [False, True])
@pytest.mark.parametrize("purged_size", [0, 1])
@pytest.mark.parametrize("train_size", [1, pd.Timedelta(days=7)])
@pytest.mark.parametrize("previous", [False, True])
def test_walk_forward_shifted_date_after_data(
    expand_train, reduce_test, purged_size, train_size, previous
):
    """A test ending beyond the data is included only when reduce_test is True."""
    X = pd.DataFrame(
        {"x": np.zeros(60)}, index=pd.date_range(end="2024-03-01", periods=60)
    )
    cv = WalkForward(
        train_size=train_size,
        test_size=1,
        freq="W-FRI",
        freq_offset=pd.Timedelta(days=1),
        previous=previous,
        expand_train=expand_train,
        reduce_test=reduce_test,
        purged_size=purged_size,
    )

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == (7 if reduce_test else 6)
    np.testing.assert_array_equal(
        np.concatenate([test for _, test in splits]),
        np.arange(11, 60 if reduce_test else 53),
    )
    for i, (train, test) in enumerate(splits):
        train_start = 0 if expand_train else 4 + 7 * i
        np.testing.assert_array_equal(
            train, np.arange(train_start, 11 + 7 * i - purged_size)
        )
        np.testing.assert_array_equal(test, np.arange(11 + 7 * i, 18 + 7 * i))


@pytest.mark.parametrize("reduce_test", [False, True])
@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("train_size", [2, pd.Timedelta(days=2)])
@pytest.mark.parametrize("expand_train,purged_size", [(False, 0), (True, 1)])
def test_walk_forward_shifted_daily_dates_respect_reduce_test(
    reduce_test, previous, train_size, expand_train, purged_size
):
    """Shifting daily dates by whole days must preserve which final tests are kept."""
    X = pd.DataFrame({"x": np.zeros(10)}, index=pd.date_range("2024-01-01", periods=10))
    params = dict(
        train_size=train_size,
        test_size=3,
        freq="D",
        previous=previous,
        expand_train=expand_train,
        purged_size=purged_size,
        reduce_test=reduce_test,
    )
    cv = WalkForward(**params, freq_offset=pd.Timedelta(days=3))
    expected = list(WalkForward(**params).split(X))
    splits = list(cv.split(X))

    assert (
        cv.get_n_splits(X) == len(splits) == len(expected) == (3 if reduce_test else 2)
    )
    for (train, test), (expected_train, expected_test) in zip(
        splits, expected, strict=True
    ):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
    if reduce_test:
        np.testing.assert_array_equal(splits[-1][1], [8, 9])


@pytest.mark.parametrize("test_size", [2**63 - 1, 2**63, 10**100])
@pytest.mark.parametrize("train_size", [2, pd.Timedelta(days=2)])
@pytest.mark.parametrize("reduce_test", [False, True])
def test_walk_forward_large_calendar_test_size(test_size, train_size, reduce_test):
    """A test size beyond NumPy's integer range must not overflow calendar indexing."""
    X = pd.DataFrame({"x": np.zeros(10)}, index=pd.date_range("2024-01-01", periods=10))
    cv = WalkForward(
        train_size=train_size, test_size=test_size, freq="D", reduce_test=reduce_test
    )

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == int(reduce_test)
    if reduce_test:
        np.testing.assert_array_equal(splits[0][0], np.arange(2))
        np.testing.assert_array_equal(splits[0][1], np.arange(2, 10))


@pytest.mark.parametrize("reduce_test", [False, True])
@pytest.mark.parametrize(
    "train_size,previous,purged_size,test_date",
    [
        (1, False, 0, "2024-01-08"),
        (1, True, 0, "2024-01-05"),
        (pd.Timedelta(days=1), True, 0, "2024-01-05"),
        (3, False, 1, "2024-01-08"),
        (pd.Timedelta(days=3), False, 1, "2024-01-08"),
    ],
)
def test_walk_forward_rejects_empty_calendar_training_before_first_split(
    reduce_test, train_size, previous, purged_size, test_date
):
    """Report a later empty training window before yielding any earlier valid fold."""
    X = pd.DataFrame(
        {"x": np.zeros(10)}, index=pd.bdate_range("2024-01-01", periods=10)
    )
    cv = WalkForward(
        train_size=train_size,
        test_size=1,
        freq="D",
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
    )
    match = f"test starting {test_date}.*contains no observations"

    with pytest.raises(ValueError, match=match) as split_error:
        next(cv.split(X))
    with pytest.raises(ValueError, match=match) as count_error:
        cv.get_n_splits(X)

    message = str(split_error.value)
    assert message == str(count_error.value)
    assert "Increase train_size" in message
    assert "freq=None" in message
    assert ("reduce purged_size" in message) == (purged_size > 0)


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("train_size", [1, pd.Timedelta(days=1)])
def test_walk_forward_expanding_training_covers_missing_calendar_periods(
    previous, train_size
):
    """Expansion can provide history when a rolling calendar window would be empty."""
    X = pd.DataFrame(
        {"x": np.zeros(10)}, index=pd.bdate_range("2024-01-01", periods=10)
    )
    cv = WalkForward(
        train_size=train_size,
        test_size=1,
        freq="D",
        previous=previous,
        expand_train=True,
    )

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == 8
    for i, (train, test) in enumerate(splits, start=1):
        np.testing.assert_array_equal(train, np.arange(i))
        np.testing.assert_array_equal(test, [i])


@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("purged_size", [3, 4, 10**100])
@pytest.mark.parametrize("train_size", [3, pd.Timedelta(days=3)])
def test_walk_forward_rejects_calendar_training_removed_by_purging(
    expand_train, purged_size, train_size
):
    """Purging must leave at least one observation, including with expanding training."""
    X = pd.DataFrame({"x": np.zeros(10)}, index=pd.date_range("2024-01-01", periods=10))
    cv = WalkForward(
        train_size=train_size,
        test_size=1,
        freq="D",
        expand_train=expand_train,
        purged_size=purged_size,
    )
    match = "test starting 2024-01-04.*contains no observations.*reduce purged_size"

    with pytest.raises(ValueError, match=match) as split_error:
        next(cv.split(X))
    with pytest.raises(ValueError, match=match) as count_error:
        cv.get_n_splits(X)
    assert str(split_error.value) == str(count_error.value)


@pytest.mark.parametrize("expand_train", [False, True])
@pytest.mark.parametrize("freq_offset", [None, pd.Timedelta(days=1)])
@pytest.mark.parametrize(
    "start,n_samples,expected_n_splits", [("2024-01-01", 6, 4), ("2024-01-05", 2, 0)]
)
def test_walk_forward_ignores_empty_training_for_omitted_test_windows(
    expand_train, freq_offset, start, n_samples, expected_n_splits
):
    """An omitted test window needs no training observations."""
    X = pd.DataFrame(
        {"x": np.zeros(n_samples)}, index=pd.bdate_range(start, periods=n_samples)
    )
    cv = WalkForward(
        train_size=1,
        test_size=1,
        freq="D",
        expand_train=expand_train,
        freq_offset=freq_offset,
    )

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == expected_n_splits
    assert all(train.size > 0 and test.size > 0 for train, test in splits)


def test_walk_forward_missing_observations_do_not_extend_calendar_windows():
    """Three calendar days may contain just one observation in either window."""
    X = pd.DataFrame({"x": np.zeros(8)}, index=pd.bdate_range("2024-01-02", periods=8))
    cv = WalkForward(train_size=3, test_size=3, freq="D")

    splits = list(cv.split(X))
    assert cv.get_n_splits(X) == len(splits) == 2
    # Friday through Sunday contains only Friday's observation (index 3).
    np.testing.assert_array_equal(splits[0][0], [0, 1, 2])
    np.testing.assert_array_equal(splits[0][1], [3])
    np.testing.assert_array_equal(splits[1][0], [3])
    np.testing.assert_array_equal(splits[1][1], [4, 5, 6])


@pytest.mark.parametrize("reduce_test", [False, True])
def test_walk_forward_negative_offset_keeps_last_rebalancing_date(reduce_test):
    """A negative `freq_offset` shifts anchors backwards, so an anchor falling
    after the data can still produce a rebalancing date inside the data."""
    index = pd.date_range("2024-01-01", "2024-01-28", freq="D")
    X = pd.DataFrame(np.zeros((len(index), 1)), index=index)
    cv = WalkForward(
        test_size=1,
        train_size=1,
        freq="W-MON",
        freq_offset=dt.timedelta(days=-3),
        reduce_test=reduce_test,
    )
    test_windows = [
        (index[test[0]].date(), index[test[-1]].date()) for _, test in cv.split(X)
    ]
    # Rebalancing dates are the Fridays 5, 12, 19 and 26 January.
    expected = [
        (dt.date(2024, 1, 5), dt.date(2024, 1, 11)),
        (dt.date(2024, 1, 12), dt.date(2024, 1, 18)),
        (dt.date(2024, 1, 19), dt.date(2024, 1, 25)),
    ]
    if reduce_test:
        expected.append((dt.date(2024, 1, 26), dt.date(2024, 1, 28)))
    assert test_windows == expected
    assert cv.get_n_splits(X) == len(expected)
