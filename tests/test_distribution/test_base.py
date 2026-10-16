from __future__ import annotations

import numpy as np
import pytest

from skfolio.distribution import BaseDistribution, Gaussian, GaussianCopula, VineCopula


def test_base_distribution_is_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseDistribution()


@pytest.mark.parametrize(
    "model,n_features",
    [
        (Gaussian(), 1),
        (GaussianCopula(), 2),
        (
            VineCopula(
                marginal_candidates=[Gaussian()],
                copula_candidates=[GaussianCopula()],
                random_state=0,
            ),
            3,
        ),
    ],
)
def test_bic_list_input_matches_array(model, n_features):
    X = np.random.default_rng(0).uniform(0.01, 0.99, (100, n_features))
    model.fit(X)
    assert model.bic(X.tolist()) == model.bic(X)
