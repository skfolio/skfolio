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


def test_drop_high_correlation_absolute():
    """Apply absolute correlations throughout the removal algorithm."""
    expected_corr = np.array([[1.0, -0.8, -0.5], [-0.8, 1.0, 0.0], [-0.5, 0.0, 1.0]])
    corr_sqrt = np.linalg.cholesky(expected_corr).T
    # Mirrored square-root rows are centered and reproduce the target correlation.
    X = np.vstack((corr_sqrt, -corr_sqrt))
    corr = np.corrcoef(X.T)
    np.testing.assert_allclose(corr, expected_corr, atol=1e-15)

    threshold = 0.75
    # Only assets 0 and 1 cross the threshold; asset 0 has the higher mean
    # absolute correlation and should therefore be removed.
    triu_idx = np.triu_indices(3, 1)
    assert np.flatnonzero(np.abs(corr)[triu_idx] > threshold).tolist() == [0]
    assert np.abs(corr).mean(axis=0)[0] > np.abs(corr).mean(axis=0)[1]

    model = DropCorrelated(threshold=threshold)
    model.fit(X)
    assert model.to_keep_.tolist() == [
        True,
        True,
        True,
    ]
    model = DropCorrelated(threshold=threshold, absolute=True)
    model.fit(X)
    assert model.to_keep_.tolist() == [False, True, True]


def test_drop_high_correlation_rejects_non_boolean_absolute():
    """Reject non-boolean values for the absolute option."""
    model = DropCorrelated().set_params(absolute="yes")
    with pytest.raises(ValueError, match="absolute must be a boolean"):
        model.fit(np.ones((2, 2)))


@pytest.mark.parametrize("threshold", [-1.5, 1.5])
def test_drop_high_correlation_invalid_threshold(threshold):
    X = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(ValueError, match="`threshold` must be between -1 and 1"):
        DropCorrelated(threshold=threshold).fit(X)
