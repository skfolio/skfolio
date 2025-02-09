import numpy as np
import sklearn as sk

from skfolio.distribution.univariate._base import BaseUnivariate


def find_best_and_fit_univariate_dist(
    X: np.ndarray, distribution_candidates: list[BaseUnivariate]
) -> BaseUnivariate:
    """Find the best marginal univariate distribution that minimize the BIC
    criterion and returned the fitted model."""

    results = []
    for dist in distribution_candidates:
        if not isinstance(dist, BaseUnivariate):
            raise ValueError(
                "The candidate distribution must inherit from BaseUnivariate"
            )
        dist = sk.clone(dist)
        dist.fit(X)
        bic = dist.bic(X)
        results.append((dist, bic))

    best_dist = min(results, key=lambda x: x[1])[0]
    return best_dist
