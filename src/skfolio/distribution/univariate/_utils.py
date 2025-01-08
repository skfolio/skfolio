import numpy as np

from skfolio.distribution.univariate._base import BaseUnivariate
from skfolio.distribution.univariate._gaussian import Gaussian
from skfolio.distribution.univariate._student_t import StudentT


def optimal_univariate_dist(
    X: np.ndarray, candidate_distributions: list[BaseUnivariate] | None = None
) -> BaseUnivariate:
    """Find the optimal marginal univariate distribution that minimize the BIC
    criterion"""
    if candidate_distributions is None:
        candidate_distributions = [Gaussian(), StudentT()]

    results = []
    for dist in candidate_distributions:
        if not isinstance(dist, BaseUnivariate):
            raise ValueError(
                "The candidate distribution must inherit from BaseUnivariate"
            )
        dist.fit(X)
        bic = dist.bic(X)
        results.append((dist, bic))

    best_dist = min(results, key=lambda x: x[1])[0]
    return best_dist
