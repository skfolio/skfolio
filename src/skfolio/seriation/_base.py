"""Base seriation estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv

from skfolio.typing import ArrayLike, BoolArray, FloatArray, IntArray, StrArray
from skfolio.utils.stats import assert_is_distance
from skfolio.utils.validation import _validate_pairwise_matrix


class BaseSeriation(skb.BaseEstimator, ABC):
    """Base class for estimators that order assets from a distance matrix.

    A NaN diagonal marks an asset as non-investable for the current ordering.
    The matrix restricted to investable assets must be finite, symmetric and
    nonnegative, with a zero diagonal, within the tolerances described below.
    DataFrame row and column labels must match in the same order.

    Attributes
    ----------
    ordering_ : ndarray of shape (n_investable_assets,)
        Positions in the original input matrix, listed in the computed order.
        Each investable asset appears exactly once. Empty when no assets are
        investable. A single investable asset produces its original position.

    investable_mask_ : ndarray of shape (n_assets,)
        Boolean mask selecting investable assets for the current ordering.

    n_features_in_ : int
        Number of assets in the full schema.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Asset names, defined when the input names are all strings.

    Notes
    -----
    Symmetry and zero diagonal checks use an absolute tolerance of 1e-5.
    Symmetry uses no relative tolerance. Negative distances down to -1e-8 are
    accepted and clipped to zero. Accepted asymmetry is averaged, and diagonal
    entries are set to zero. The input matrix is not modified.
    """

    ordering_: IntArray
    investable_mask_: BoolArray
    n_features_in_: int
    feature_names_in_: StrArray

    @abstractmethod
    def fit(self, X: ArrayLike, y: None = None) -> BaseSeriation:
        """Fit an ordering from a distance snapshot.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Distance matrix. NaN diagonal entries mark non-investable assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BaseSeriation
            Fitted estimator.
        """
        ...

    def _validate_distance(
        self, X: ArrayLike, *, reset: bool
    ) -> tuple[FloatArray, BoolArray]:
        """Validate the input and return distances between investable assets.

        Apply the numerical corrections described in the class documentation
        without modifying the input matrix.

        Parameters
        ----------
        X : array-like of shape (n_assets, n_assets)
            Distance matrix. NaN diagonal entries mark non-investable assets.

        reset : bool
            If True, initialize the feature count and names from `X`.
            If False, validate against the existing feature count and names.

        Returns
        -------
        distance : ndarray of shape (n_investable_assets, n_investable_assets)
            Distances between investable assets in their original input order.
            Symmetric and nonnegative, with a zero diagonal.

        investable_mask : ndarray of bool of shape (n_assets,)
            Investable asset mask covering the full input matrix.

        Raises
        ------
        ValueError
            If the distance matrix or asset labels are invalid, or the input
            conflicts with the established asset schema when `reset=False`.
        """
        distance, investable_mask = _validate_pairwise_matrix(X)
        distance = distance[np.ix_(investable_mask, investable_mask)]
        assert_is_distance(distance)
        if np.any(distance < -1e-8):
            raise ValueError("Distances must be nonnegative.")
        distance = np.maximum(distance / 2 + distance.T / 2, 0)
        np.fill_diagonal(distance, 0)
        skv.validate_data(self, X, reset=reset, skip_check_array=True)
        return distance, investable_mask

    def __sklearn_tags__(self) -> sku.Tags:
        """Declare pairwise inputs with unavailable assets represented by NaN."""
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = True
        tags.input_tags.allow_nan = True
        return tags
