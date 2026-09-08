from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.pre_selection import SelectNonDominated


def test_select_non_dominated(X):
    with config_context(transform_output="pandas"):
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


def test_select_non_dominated_rejects_invalid_threshold():
    model = SelectNonDominated(threshold=1.1)

    with pytest.raises(ValueError, match="between -1 and 1"):
        model.fit(np.ones((5, 2)))


def test_select_non_dominated_keeps_small_universe():
    X = np.arange(15, dtype=float).reshape(5, 3)
    model = SelectNonDominated(min_n_assets=3)

    model.fit(X)

    np.testing.assert_array_equal(model.get_support(), [True, True, True])


def test_select_non_dominated_considers_negatively_correlated_pair():
    z = np.linspace(-1.0, 1.0, 9)
    X = np.column_stack([0.01 + 0.01 * z, -0.02 * z])

    with_pairs = SelectNonDominated(threshold=0.0).fit(X)
    without_pairs = SelectNonDominated(threshold=-1.0).fit(X)

    np.testing.assert_array_equal(with_pairs.get_support(), [True, True])
    np.testing.assert_array_equal(without_pairs.get_support(), [True, False])
