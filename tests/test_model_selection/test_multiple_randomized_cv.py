"""Test Multiple Randomized CV."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from skfolio import Population, RatioMeasure
from skfolio.model_selection import (
    MultipleRandomizedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.optimization import InverseVolatility
from skfolio.pre_selection import SelectKExtremes


def assert_split_equal(split, res):
    for i, (train, test, assets) in enumerate(split):
        assert np.array_equal(train, res[i][0])
        assert np.array_equal(test, res[i][1])
        assert np.array_equal(assets, res[i][2])


def test_invalid_init():
    X = np.random.randn(60, 10)

    # walk_forward not correct type
    with pytest.raises(
        TypeError, match="`walk_forward` must be a `WalkForward` instance"
    ):
        cv = MultipleRandomizedCV(
            walk_forward=object(), n_subsamples=1, asset_subset_size=1
        )
        list(cv.split(X))

    # invalid num_subsamples
    wf = WalkForward(test_size=30, train_size=252)
    with pytest.raises(ValueError, match="n_subsample=0 must satisfy"):
        cv = MultipleRandomizedCV(walk_forward=wf, n_subsamples=0, asset_subset_size=1)
        list(cv.split(X))

    with pytest.raises(ValueError, match="n_subsample=1000 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, n_subsamples=1000, asset_subset_size=1
        )
        list(cv.split(X))

    # invalid asset_subset_size
    with pytest.raises(ValueError, match="asset_subset_size=0 must satisfy"):
        cv = MultipleRandomizedCV(walk_forward=wf, n_subsamples=2, asset_subset_size=0)
        list(cv.split(X))

    with pytest.raises(ValueError, match="asset_subset_size=10 must satisfy"):
        cv = MultipleRandomizedCV(walk_forward=wf, n_subsamples=2, asset_subset_size=10)
        list(cv.split(X))

    # invalid window_size
    with pytest.raises(ValueError, match="When not None, window_size=0 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, n_subsamples=2, asset_subset_size=2, window_size=0
        )
        list(cv.split(X))
    with pytest.raises(ValueError, match="When not None, window_size=61 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, n_subsamples=2, asset_subset_size=2, window_size=61
        )
        list(cv.split(X))

    # invalid WalkForward vs window_size
    with pytest.raises(ValueError, match="The sum of"):
        cv = MultipleRandomizedCV(walk_forward=wf, n_subsamples=10, asset_subset_size=2)
        list(cv.split(X))


def test_get_n_splits_and_get_path_ids_before_split():
    wf = WalkForward(test_size=30, train_size=252)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        n_subsamples=3,
        asset_subset_size=2,
        window_size=None,
        random_state=0,
    )
    # get_path_ids before split raises
    with pytest.raises(ValueError, match="Before get_path_ids you must call split"):
        cv.get_path_ids()


def test_split_without_window_size_1():
    X = np.random.randn(10, 20)
    wf = WalkForward(test_size=2, train_size=3)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        n_subsamples=3,
        asset_subset_size=2,
        random_state=42,
    )
    splits = list(cv.split(X))
    assert len(splits) == 9
    # each element is a tuple of (train, test, assets)
    for _, (train, test, assets) in enumerate(splits):
        # train/test come from dummy logic
        assert train.shape == (3,)
        assert test.shape == (2,)
        # assets length matches asset_subset_size
        assert isinstance(assets, np.ndarray)
        assert assets.shape == (2,)
        # all asset indices are in valid range
        assert np.all((assets >= 0) & (assets < X.shape[1]))
    # now get_path_ids matches subsample ids
    path_ids = cv.get_path_ids()
    assert np.array_equal(path_ids, [0, 0, 0, 1, 1, 1, 2, 2, 2])


def test_split_without_window_size_2():
    X = np.random.randn(6, 5)
    wf = WalkForward(test_size=1, train_size=2)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        n_subsamples=2,
        asset_subset_size=3,
        random_state=0,
    )
    assert_split_equal(
        cv.split(X),
        [
            ([0, 1], [2], [0, 1, 4]),
            ([1, 2], [3], [0, 1, 4]),
            ([2, 3], [4], [0, 1, 4]),
            ([3, 4], [5], [0, 1, 4]),
            ([0, 1], [2], [1, 3, 4]),
            ([1, 2], [3], [1, 3, 4]),
            ([2, 3], [4], [1, 3, 4]),
            ([3, 4], [5], [1, 3, 4]),
        ],
    )

    assert np.array_equal(cv.get_path_ids(), [0, 0, 0, 0, 1, 1, 1, 1])


def test_split_with_window_size():
    X = np.random.randn(20, 10)
    wf = WalkForward(test_size=2, train_size=3)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        n_subsamples=2,
        asset_subset_size=4,
        window_size=8,
        random_state=0,
    )
    assert_split_equal(
        cv.split(X),
        [
            ([5, 6, 7], [8, 9], [0, 1, 3, 9]),
            ([7, 8, 9], [10, 11], [0, 1, 3, 9]),
            ([0, 1, 2], [3, 4], [0, 6, 7, 8]),
            ([2, 3, 4], [5, 6], [0, 6, 7, 8]),
        ],
    )

    assert np.array_equal(cv.get_path_ids(), [0, 0, 1, 1])


def test_time_aware_wf(X):
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(test_size=3, train_size=12, freq="WOM-3FRI"),
        window_size=252 * 2,
        asset_subset_size=5,
        n_subsamples=100,
        random_state=1,
    )
    splits = list(cv.split(X))
    assert len(splits) == 309

    for split in splits:
        assert 247 <= len(split[0]) <= 258
        assert 56 <= len(split[1]) <= 69
        assert len(split[2]) == 5
        assert np.all((split[2] >= 0) & (split[2] < X.shape[1]))

    path_ids = cv.get_path_ids()
    assert len(path_ids) == 309
    assert path_ids.min() == 0
    assert path_ids.max() == 99


def test_time_cross_val_predict(X):
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(test_size=3, train_size=12, freq="WOM-3FRI"),
        window_size=252 * 2,
        asset_subset_size=5,
        n_subsamples=100,
        random_state=1,
    )

    model = InverseVolatility()
    pred = cross_val_predict(model, X, cv=cv)
    assert isinstance(pred, Population)
    assert pred.plot_composition()
    assert pred.plot_cumulative_returns()
    assert pred.plot_distribution(measure_list=[RatioMeasure.SHARPE_RATIO])
    assert pred[0].plot_weights_per_observation()

    model = Pipeline(
        [("pre_selection", SelectKExtremes(k=10)), ("allocation", InverseVolatility())]
    )

    pred = cross_val_predict(model, X, cv=cv)
    assert isinstance(pred, Population)
    assert len(pred) == 100
