"""Base Feature Extraction Estimators."""

# Copyright (c) 2026
# Author: Ahmed Nabil Atwa
# SPDX-License-Identifier: BSD-3-Clause
# Wraps gen_fex (Apache-2.0) with attribution

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb


class BaseFeatureExtractor(skb.BaseEstimator, ABC):
    """
    Base class for all feature extractors in skfolio.

    Feature extractors perform dimensionality reduction while
    preserving statistical properties important for portfolio optimization.

    Notes
    -----
    All estimators should specify all parameters in __init__ as
    explicit keyword arguments (no *args or **kwargs).
    """

    load_matrix_: np.ndarray
    mean_vector_: np.ndarray

    @abstractmethod
    def __init__(self):
        """Initialize the feature extractor."""

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None) -> "BaseFeatureExtractor":
        """Fit the feature extractor."""

    @abstractmethod
    def transform(self, X: npt.ArrayLike | None = None) -> np.ndarray:
        """Apply dimensionality reduction."""

    @abstractmethod
    def inverse_transform(self, X_transformed: npt.ArrayLike) -> np.ndarray:
        """Transform data back to original space."""
