import numpy as np
from sklearn import set_config

from skfolio import PerfMeasure
from skfolio.pre_selection import SelectKExtremes


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
