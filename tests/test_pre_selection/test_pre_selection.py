import numpy as np
import pytest
from sklearn import set_config

from skfolio import PerfMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.pre_selection import DropCorrelated, SelectKExtremes, SelectNonDominated
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    return X


def test_drop_high_correlation(X):
    set_config(transform_output="pandas")

    model = DropCorrelated(threshold=0.5)
    model.fit(X)
    assert np.all(
        model.feature_names_in_[model.to_keep_] == model.get_feature_names_out()
    )
    new_X = model.transform(X)
    assert new_X.shape[0] == X.shape[0]
    assert new_X.shape[1] < X.shape[1]
    assert np.all(new_X.columns == model.feature_names_in_[model.to_keep_])

    model = DropCorrelated(threshold=0.5)
    new_X = model.fit_transform(X)
    assert new_X.shape[1] < X.shape[1]

    corr = new_X.corr().to_numpy()
    assert np.all(corr[np.triu_indices(corr.shape[1], 1)] < 0.5)

    new_new_X = model.fit_transform(new_X)
    assert new_new_X.shape == new_X.shape


def test_select_k_extremes(X):
    set_config(transform_output="pandas")

    model = SelectKExtremes(k=10)
    model.fit(X)
    assert np.all(
        model.feature_names_in_[model.to_keep_] == model.get_feature_names_out()
    )
    new_X = model.transform(X)
    assert new_X.shape[0] == X.shape[0]
    assert new_X.shape[1] == 10
    assert np.all(new_X.columns == model.feature_names_in_[model.to_keep_])

    means = np.asarray(np.mean(X, axis=0))

    model = SelectKExtremes(k=10, measure=PerfMeasure.MEAN)
    new_X = model.fit_transform(X)
    highest = X.columns[np.argsort(means)][-10:]
    assert set(new_X.columns) == set(highest)

    model = SelectKExtremes(k=10, measure=PerfMeasure.MEAN, highest=False)
    new_X = model.fit_transform(X)
    lowest = X.columns[np.argsort(means)][:10]
    assert set(new_X.columns) == set(lowest)


def test_select_non_dominated(X):
    set_config(transform_output="pandas")

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
