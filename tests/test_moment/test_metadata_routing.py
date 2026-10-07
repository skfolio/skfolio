"""Default activity requests through batch and online metadata routers."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from skfolio.model_selection import online_score
from skfolio.moments import (
    EWCovariance,
    EWMu,
    EWVariance,
    RegimeAdjustedEWCovariance,
    RegimeAdjustedEWVariance,
)


@pytest.mark.parametrize(
    "estimator,attribute",
    [
        (EWMu, "mu_"),
        (EWCovariance, "covariance_"),
        (EWVariance, "variance_"),
        (RegimeAdjustedEWCovariance, "covariance_"),
        (RegimeAdjustedEWVariance, "variance_"),
    ],
)
@pytest.mark.parametrize("metadata_name", ["active_mask", "membership"])
def test_activity_routing_in_pipeline_and_online_score(
    estimator, attribute, metadata_name
):
    X = np.random.default_rng(21).normal(0, 0.01, (50, 4))
    active = np.ones(X.shape, dtype=bool)
    active[20:30, 3] = False

    def count_investable(model, X, y=None):
        moments = getattr(model, attribute)
        if moments.ndim == 2:
            moments = np.diag(moments)
        return np.count_nonzero(np.isfinite(moments))

    with config_context(enable_metadata_routing=True):
        model = estimator(half_life=5, min_observations=3)
        if metadata_name != "active_mask":
            model.set_fit_request(active_mask=metadata_name)
            model.set_partial_fit_request(active_mask=metadata_name)
        pipeline = make_pipeline(clone(model))
        pipeline.fit(X[:30], **{metadata_name: active[:30]})
        direct = clone(model).fit(X[:30], active_mask=active[:30])
        np.testing.assert_allclose(
            getattr(pipeline[-1], attribute), getattr(direct, attribute), equal_nan=True
        )
        assert count_investable(pipeline[-1], X) == 3
        scores = online_score(
            model,
            X,
            warmup_size=20,
            test_size=10,
            scoring=count_investable,
            params={metadata_name: active},
            per_step=True,
        )
    np.testing.assert_array_equal(scores, [4, 3, 4])
