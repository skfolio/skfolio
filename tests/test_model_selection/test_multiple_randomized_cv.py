"""Test Multiple Randomized CV."""

import numpy as np
import pytest
from skfolio import RatioMeasure
from skfolio.model_selection import (
    MultipleRandomizedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.optimization import InverseVolatility


def test_invalid_init():
    X = np.random.randn(60, 10)

    # walk_forward not correct type
    with pytest.raises(TypeError, match="`walk_forward` must be a `WalkForward` instance"):
        cv = MultipleRandomizedCV(
            walk_forward=object(), num_subsamples=1, asset_subset_size=1
        )
        list(cv.split(X))

    # invalid num_subsamples
    wf =  WalkForward(test_size=30, train_size=252)
    with pytest.raises(ValueError,  match="n_subsample=0 must satisfy"):
        cv = MultipleRandomizedCV(walk_forward=wf, num_subsamples=0, asset_subset_size=1)
        list(cv.split(X))

    with pytest.raises(ValueError, match="n_subsample=1000 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, num_subsamples=1000, asset_subset_size=1
        )
        list(cv.split(X))

    # invalid asset_subset_size
    with pytest.raises(ValueError, match="asset_subset_size=0 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, num_subsamples=2, asset_subset_size=0
        )
        list(cv.split(X))

    with pytest.raises(ValueError, match="asset_subset_size=10 must satisfy"):
        cv = MultipleRandomizedCV(
            walk_forward=wf, num_subsamples=2, asset_subset_size=10
        )
        list(cv.split(X))


    # invalid window_size
    with pytest.raises(ValueError, match="When not None, window_size=0 must satisfy"):
        cv =        MultipleRandomizedCV(
            walk_forward=wf, num_subsamples=2, asset_subset_size=2, window_size=0
        )
        list(cv.split(X))
    with pytest.raises(ValueError, match="When not None, window_size=61 must satisfy"):
        cv =        MultipleRandomizedCV(
            walk_forward=wf, num_subsamples=2, asset_subset_size=2, window_size=61
        )
        list(cv.split(X))

    # invalid WalkForward vs window_size
    with pytest.raises(ValueError, match="The sum of"):
        cv =        MultipleRandomizedCV(
        walk_forward=wf, num_subsamples=10, asset_subset_size=2
        )
        list(cv.split(X))


def test_get_n_splits_and_get_path_ids_before_split():
    wf = WalkForward(test_size=30, train_size=252)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        num_subsamples=3,
        asset_subset_size=2,
        window_size=None,
        random_state=0,
    )
    # get_path_ids before split raises
    with pytest.raises(ValueError, match="Before get_path_ids you must call split"):
        cv.get_path_ids()


def test_split_without_window_size():
    X = np.random.randn(10, 20)
    wf =  WalkForward(test_size=2, train_size=3)
    cv = MultipleRandomizedCV(
        walk_forward=wf,
        num_subsamples=3,
        asset_subset_size=2,
        random_state=42,
    )
    splits = list(cv.split(X))
    assert len(splits) == 9
    # each element is a tuple of (train, test, assets)
    for idx, (train, test, assets) in enumerate(splits):
        # train/test come from dummy logic
        assert train.shape == (2,)
        assert test.shape == (1,)
        # assets length matches asset_subset_size
        assert isinstance(assets, np.ndarray)
        assert assets.shape == (2,)
        # all asset indices are in valid range
        assert np.all((assets >= 0) & (assets < X.shape[1]))
    # now get_path_ids matches subsample ids
    path_ids = cv.get_path_ids()
    # since dummy n_splits=2, subsample id repeats twice
    expected = np.repeat(np.arange(3), 2)
    assert np.array_equal(path_ids, expected)


def test_split_with_window_size():
    dummy = DummyWalkForward(n_splits=1, train_size=2, test_size=1)
    window = 3
    cv = MultipleRandomizedCV(
        walk_forward=dummy,
        num_subsamples=2,
        asset_subset_size=2,
        window_size=window,
        random_state=0,
    )
    splits = list(cv.split(X))
    # total splits = 2*1 =2
    assert len(splits) == 2
    for subsample_id, (train, test, assets) in enumerate(splits):
        # indices within [start, start+window)
        assert train.max() < 6
        assert test.max() < 6
        # span of indices should not exceed window
        assert max(train.max(), test.max()) - min(train.min(), test.min()) < window


def test_asset_subset_uniqueness_and_range():
    dummy = DummyWalkForward(n_splits=1, train_size=1, test_size=1)
    cv = MultipleRandomizedCV(
        walk_forward=dummy,
        num_subsamples=5,
        asset_subset_size=3,
        window_size=None,
        random_state=1,
    )
    splits = list(cv.split(X))
    # extract asset subsets per subsample id
    path_ids = cv.get_path_ids()
    subsets = {pid: assets.tolist() for (_, _, assets), pid in zip(splits, path_ids)}
    # ensure all subsets are unique
    all_sets = [tuple(sorted(sub)) for sub in subsets.values()]
    assert len(set(all_sets)) == 5


def test_multiple_randomized_cv():
    X = np.random.randn(60, 10)

    walk_forward = WalkForward(test_size=5, train_size=10)
    cv = MultipleRandomizedCV(
        walk_forward=walk_forward,
        n_sample_observations=50,
        n_sample_assets=5,
        n_subsamples=10,
        random_state=1,
    )
    list(cv.split(X))

    cv.get_path_ids()


def test_multiple_randomized_cv_df(X):
    walk_forward = WalkForward(test_size=30, train_size=252)
    cv = MultipleRandomizedCV(
        walk_forward=walk_forward,
        n_sample_observations=252 * 2,
        n_sample_assets=5,
        n_subsamples=100,
        random_state=1,
    )
    list(cv.split(X))

    cv.get_path_ids()

    model = InverseVolatility()
    pred = cross_val_predict(model, X, cv=cv)
    pred.plot_composition().show()
    pred.plot_cumulative_returns().show()
    pred.plot_distribution(measure_list=[RatioMeasure.SHARPE_RATIO]).show()
    pred[0].plot_weights_per_observation().show()
