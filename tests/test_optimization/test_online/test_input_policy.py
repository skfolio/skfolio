"""Input failures must have explicit effects on a live online stream."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1, -1.2])
@pytest.mark.parametrize(
    "factory", [FollowTheWinner, lambda: FollowTheLoser(strategy="pamr")]
)
def test_invalid_block_does_not_consume_valid_prefix(invalid, factory):
    estimator = factory().partial_fit([[0.1, 0.02]])
    previous = estimator.weights_.copy()
    count = estimator.n_observations_
    with pytest.raises(ValueError):
        estimator.partial_fit([[0.02, 0.01], [invalid, 0.03]])
    assert_allclose(estimator.weights_, previous)
    assert estimator.n_observations_ == count
    estimator.partial_fit([[0.01, 0.02]])
    assert estimator.n_observations_ == count + 1


def test_asset_names_are_fixed_until_explicit_reset():
    estimator = FollowTheWinner().fit(pd.DataFrame([[0.1, 0.2]], columns=["a", "b"]))
    with pytest.raises(ValueError, match="feature names"):
        estimator.partial_fit(pd.DataFrame([[0.1, 0.2]], columns=["b", "a"]))
    estimator.fit(pd.DataFrame([[0.1, 0.2, 0.3]], columns=["c", "d", "e"]))
    assert estimator.n_features_in_ == 3
    assert estimator.n_observations_ == 1


def test_named_bounds_and_groups_are_aligned():
    estimator = FollowTheWinner(
        max_weights={"a": 0.3, "b": 0.8, "c": 0.8},
        groups={"a": ["sector"], "b": ["sector"], "c": ["other"]},
        linear_constraints=["sector <= 0.6"],
    ).fit(pd.DataFrame([[0.9, 0.1, 0.01]], columns=["a", "b", "c"]))
    assert estimator.weights_[0] <= 0.3 + 1e-7
    assert estimator.weights_[:2].sum() <= 0.6 + 1e-7
