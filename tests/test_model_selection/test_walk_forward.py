from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skfolio import MultiPeriodPortfolio
from skfolio.model_selection import WalkForward, cross_val_predict
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
