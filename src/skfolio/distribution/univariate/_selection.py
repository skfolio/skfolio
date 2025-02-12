"""Univariate Distribution Selection"""

# Copyright (c) 2025
# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn as sk

from skfolio.distribution.univariate._base import BaseUnivariateDist


def select_univariate_dist(
    X: npt.ArrayLike,
    distribution_candidates: list[BaseUnivariateDist],
    aic: bool = True,
) -> BaseUnivariateDist:
    """Select the optimal univariate distribution estimator based on an information
    criterion.

    For each candidate distribution, the function fits the distribution to X and then
    computes either the Akaike Information Criterion (AIC) or the Bayesian Information
    Criterion (BIC). The candidate with the lowest criterion value is returned.

    Parameters
    ----------
    X : array-like of shape (n_observations, 1)
        The input data used to fit each candidate distribution.

    distribution_candidates : list of BaseUnivariateDist
        A list of candidate distribution estimators. Each candidate must be an instance
        of a class that inherits from `BaseUnivariateDist`.

    aic : bool, default=True
        If True, the Akaike Information Criterion (AIC) is used for model selection;
        otherwise, the Bayesian Information Criterion (BIC) is used.

    Returns
    -------
    BaseUnivariateDist
        The fitted candidate estimator that minimizes the selected information
        criterion.

    Raises
    ------
    ValueError
        If X does not have exactly one column or if any candidate in the list does not
        inherit from BaseUnivariateDist.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("X must contains one column for Univariate Distribution")

    results = {}
    for dist in distribution_candidates:
        if not isinstance(dist, BaseUnivariateDist):
            raise ValueError("Each candidate must inherit from `BaseUnivariateDist`")
        dist = sk.clone(dist)
        dist.fit(X)
        results[dist] = dist.aic(X) if aic else dist.bic(X)
    selected_dist = min(results, key=results.get)
    # noinspection PyTypeChecker
    return selected_dist
