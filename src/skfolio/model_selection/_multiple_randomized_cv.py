"""Multiple Randomized CV."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credit: Daniel Palomar
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import sklearn.utils as sku

from skfolio.model_selection._walk_forward import WalkForward
from skfolio.utils.stats import sample_unique_subsets
from skfolio.utils.tools import safe_split


class MultipleRandomizedCV:
    """Multiple Randomized Cross-Validation.

    Based on the "Multiple Randomized Backtests" methodology of Palomar & Zhou [1],
    this cross-validation strategy performs a true Monte Carlo-style evaluation by
    repeatedly sampling **contiguous** time windows and **distinct** asset subsets
    (without replacement), then applying an inner walk-forward split to each
    subsample, capturing both temporal and cross-sectional variability in performance.

    On each of `num_subsamples` iterations:
      1. Randomly pick a contiguous time window of length `window_size` (or the full history if None).
      2. Randomly pick an asset subset of size `asset_subset_size` (without replacement).
      3. Run a walk-forward split (via the supplied `walk_forward` object) on that sub-dataset.
      4. Yield `(train_indices, test_indices, asset_indices)` for each inner split.

    Parameters
    ----------
    walk_forward : WalkForward
        A :class:`~skfolio.model_selection.WalkForward` CV object to be applied to
        each subsample.

    num_subsamples : int
        Number of independent sub-datasets to draw. Each sub-dataset is a
        (time window x asset subset) on which you run the inner walk-forward.

    asset_subset_size : int
        How many assets to include in each sub-dataset.
        Must be less or equal to the total number of assets.

    window_size : int or None, default=None
        Length of the contiguous time slice (number of observations) for each
        sub-dataset. If None, uses the full time series observations in every draw.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    References
    ----------
    .. [1]  "Portfolio Optimization, Theory and Application",
        Daniel P. Palomar (2025)

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.model_selection import WalkForward, MultipleRandomizedCV
    >>> X = np.random.randn(6, 2)
    >>> cv = WalkForward(test_size=1, train_size=2)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1]
      Test:  index=[2]
    Fold 1:
      Train: index=[1 2]
      Test:  index=[3]
    Fold 2:
      Train: index=[2 3]
      Test:  index=[4]
    Fold 3:
      Train: index=[3 4]
      Test:  index=[5]
   
    """

    if TYPE_CHECKING:
        _path_ids: list[int]

    def __init__(
        self,
        walk_forward: WalkForward,
        num_subsamples: int,
        asset_subset_size: int,
        window_size: int | None = None,
        random_state: int | None = None,
    ):
        self.walk_forward = walk_forward
        self.num_subsamples = num_subsamples
        self.asset_subset_size = asset_subset_size
        self.window_size = window_size
        self.random_state = random_state

    def split(
        self, X: npt.ArrayLike, y=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y = sku.indexable(X, y)
        n_observations, n_assets = X.shape
        rng = sku.check_random_state(self.random_state)

        if not isinstance(self.walk_forward, WalkForward):
            raise TypeError("`walk_forward` must be a `WalkForward` instance")

        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise ValueError("`window_size` must be an integer")
            if self.window_size < 2  or self.window_size > n_observations:
                raise ValueError(
                    f"When not None, window_size={self.window_size} must "
                    "satisfy 2 <= window_size <= total number of observations"
                    f"={n_observations}."
                )

        if not isinstance(self.asset_subset_size, int):
            raise TypeError("`asset_subset_size` must be an integer")
        if self.asset_subset_size < 1  or self.asset_subset_size >= n_assets:
            raise ValueError(
                f"asset_subset_size={self.asset_subset_size} must satisfy 1 <= asset_subset_size < "
                f"total number of assets={n_assets}."
            )

        if not isinstance(self.num_subsamples, int):
            raise TypeError("`num_subsamples` must be an integer")
        max_num_subsamples = math.comb(n_assets, self.asset_subset_size)
        if self.num_subsamples < 2 or self.num_subsamples > max_num_subsamples:
            raise ValueError(
                f"n_subsample={self.num_subsamples} must satisfy 2 <= n_subsample <= "
                f"C({n_assets},{self.asset_subset_size})={max_num_subsamples}."
            )

        asset_indices = sample_unique_subsets(
            n=n_assets,
            k=self.asset_subset_size,
            n_subsets=self.num_subsamples,
            random_state=self.random_state,
        )

        self._path_ids = []
        for i in range(self.num_subsamples):
            if self.window_size is None:
                start_obs = 0
                X_sample = X
            else:
                start_obs = rng.integers(
                    low=0, high=n_observations - self.window_size
                )
                obs_indices = np.arange(
                    start_obs, start_obs + self.window_size
                )
                X_sample, _ = safe_split(X, indices=obs_indices, axis=0)

            for train_indices, test_indices in self.walk_forward.split(X_sample):
                self._path_ids.append(i)
                yield (
                    train_indices + start_obs,
                    test_indices + start_obs,
                    asset_indices[i, :],
                )

    def get_path_ids(self) -> np.ndarray:
        """Return the path id of each test sets in each split."""
        if not hasattr(self, "_path_ids"):
            raise ValueError("Before get_path_ids you must call split")
        return np.array(self._path_ids)
