"""Test Multiple Randomized CV."""

import numpy as np

from skfolio.model_selection import MultipleRandomizedCV, WalkForward


def test_multiple_randomized_cv():
    X = np.random.randn(60, 10)

    walk_forward = WalkForward(test_size=5, train_size=10)
    cv = MultipleRandomizedCV(
        walk_forward=walk_forward,
        n_sample_observations=50,
        n_sample_assets=5,
        n_subsamples=10,
        random_state=1,
    )
    cv.split(X)
