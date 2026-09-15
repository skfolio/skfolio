from __future__ import annotations

import numpy as np
import pytest

from skfolio.preprocessing import (
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)


@pytest.fixture(
    params=[
        CSStandardScaler(),
        CSGaussianRankScaler(),
        CSPercentileRankScaler(),
        CSTanhShrinker(),
        CSWinsorizer(),
    ]
)
def estimator(request):
    return request.param


def _clone_estimator(estimator):
    return estimator.__class__(**estimator.get_params())


class TestStatelessContract:
    """Stateless scikit-learn contract shared by cross-sectional transformers."""

    def test_transform_before_fit_matches_fit_transform(self, estimator):
        X = np.array([[1.0, 2.0, 4.0, 10.0], [4.0, 1.0, 3.0, 7.0]])

        transformed_direct = _clone_estimator(estimator).transform(X.copy())
        transformed_fit = _clone_estimator(estimator).fit_transform(X.copy())

        np.testing.assert_allclose(transformed_direct, transformed_fit, rtol=1e-12)

    def test_sklearn_tags_declare_stateless_behavior(self, estimator):
        tags = estimator.__sklearn_tags__()

        assert tags.requires_fit is False
        assert tags.input_tags.allow_nan is True


class TestFitValidation:
    """Constructor parameter validation exercised through fit."""

    @pytest.mark.parametrize(
        ("estimator", "shape", "match"),
        [
            (CSStandardScaler(min_group_size=0), (1, 4), "min_group_size"),
            (CSStandardScaler(atol=-1.0), (1, 4), "atol"),
            (CSGaussianRankScaler(min_group_size=0), (1, 4), "min_group_size"),
            (CSGaussianRankScaler(atol=-1.0), (1, 4), "atol"),
            (CSPercentileRankScaler(min_group_size=0), (1, 4), "min_group_size"),
            (CSTanhShrinker(knee=0.0), (1, 5), "knee"),
            (CSTanhShrinker(atol=-1.0), (1, 5), "atol"),
            (CSWinsorizer(low=-0.1, high=0.8), (1, 5), "low"),
            (CSWinsorizer(low=0.8, high=0.2), (1, 5), "low"),
        ],
    )
    def test_fit_validates_parameters(self, estimator, shape, match):
        with pytest.raises(ValueError, match=match):
            estimator.fit(np.ones(shape))
