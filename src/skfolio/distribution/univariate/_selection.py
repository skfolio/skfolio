"""Univariate Distribution Selection."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn as sk

from skfolio.distribution._base import SelectionCriterion
from skfolio.distribution.univariate._base import BaseUnivariateDist
from skfolio.distribution.univariate._gaussian import Gaussian
from skfolio.distribution.univariate._johnson_su import JohnsonSU
from skfolio.distribution.univariate._student_t import StudentT


def select_univariate_dist(
    X: npt.ArrayLike,
    distribution_candidates: list[BaseUnivariateDist] | None = None,
    selection_criterion: SelectionCriterion = SelectionCriterion.AIC,
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
        If None, defaults to `[Gaussian(), StudentT(), JohnsonSU()]`.

    selection_criterion : SelectionCriterion, default=SelectionCriterion.AIC
        The criterion used for model selection. Possible values are:
            - SelectionCriterion.AIC : Akaike Information Criterion
            - SelectionCriterion.BIC : Bayesian Information Criterion

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
    if distribution_candidates is None:
        distribution_candidates = [
            Gaussian(),
            StudentT(),
            JohnsonSU(),
        ]

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("X must contains one column for Univariate Distribution")

    results = {}
    for dist in distribution_candidates:
        if not isinstance(dist, BaseUnivariateDist):
            raise ValueError("Each candidate must inherit from `BaseUnivariateDist`")
        dist = sk.clone(dist)
        dist.fit(X)

        match selection_criterion:
            case selection_criterion.AIC:
                results[dist] = dist.aic(X)
            case selection_criterion.BIC:
                results[dist] = dist.bic(X)
            case _:
                raise ValueError(f"{selection_criterion} not implemented")

    selected_dist = min(results, key=results.get)
    return selected_dist
