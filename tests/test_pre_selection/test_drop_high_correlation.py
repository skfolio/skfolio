from __future__ import annotations

import numpy as np
import pytest
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


@pytest.mark.parametrize("threshold", [-1.5, 1.5])
def test_drop_high_correlation_invalid_threshold(threshold):
    X = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(ValueError, match="`threshold` must be between -1 and 1"):
        DropCorrelated(threshold=threshold).fit(X)
