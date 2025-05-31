import numpy as np
from sklearn import config_context

from skfolio.pre_selection import DropCorrelated


def test_drop_high_correlation(X):
    with config_context(transform_output="pandas"):
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
