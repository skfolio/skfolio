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
    r"""Multiple Randomized Cross-Validation.

    Based on the "Multiple Randomized Backtests" methodology of Palomar [1]_,
    this cross-validation strategy performs a resampling-based evaluation by repeatedly
    sampling **distinct** asset subsets (without replacement) and **contiguous** time
    windows, then applying an inner walk-forward split to each subsample, capturing both
    temporal and cross-sectional variability in performance.

    On each of the `n_subsamples` iterations, the following actions are performed:

    1. Randomly pick a contiguous time window of length `window_size` (or the full history if None).
    2. Randomly pick an asset subset of size `asset_subset_size` (without replacement).
    3. Run a walk-forward split (via the supplied `walk_forward` object) on that sub-dataset.
    4. Yield `(train_indices, test_indices, asset_indices)` for each inner split.

    Each asset subset is sampled without replacement (assets within each subset are
    distinct) and no subset is repeated across the `n_subsamples` draws. We employ the
    combinatorial unranking algorithm to compute any k-combination in
    `O(n_subsamples * asset_subset_size)` time and space, without generating or storing
    all :math:`M=\binom{n\_assets}{asset\_subset\_size}` subsets. When :math:`M` is
    small, this guarantees exhaustive coverage of every possible asset-universe.
    Because ranks are drawn without replacement from a finite population of size
    :math:`M`, the variance of the sample mean is reduced by the finite-population
    correction factor :math:`\tfrac{M - n\_subsamples}{M - 1}`.

    Parameters
    ----------
    walk_forward : WalkForward
        A :class:`~skfolio.model_selection.WalkForward` CV object to be applied to
        each subsample.

    n_subsamples : int
        Number of independent subsamples (sub-datasets) to draw. Each subsample is a
        (time window x asset subset) on which you run the inner walk-forward.

    asset_subset_size : int
        How many assets to include in each subsample.
        Must be less or equal to the total number of assets.

    window_size : int or None, default=None
        Length of the contiguous time slice (number of observations) for each
        subsample. If None, uses the full time series observations in every draw.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    References
    ----------
    .. [1]  "Portfolio Optimization: Theory and Application", Chapter 8,
            Daniel P. Palomar (2025)

    Examples
    --------
    Tutorials using `MultipleRandomizedCV`:
        * :ref:`sphx_glr_auto_examples_model_selection_plot_1_multiple_randomized_cv.py`


    >>> import numpy as np
    >>> from skfolio.datasets import load_sp500_dataset, load_factors_dataset
    >>> from skfolio.model_selection import WalkForward, MultipleRandomizedCV
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> X = np.random.randn(4, 5) # 4 observations and 5 assets.
    >>> # Draw 2 subsamples (sub-datasets) with 3 assets chosen randomly among the 5.
    >>> # For each subsample, run a Walk Forward.
    >>> # Use the full time series (no time resampling).
    >>> cv = MultipleRandomizedCV(
    ...     walk_forward=WalkForward(test_size=1, train_size=2),
    ...     n_subsamples=2,
    ...     asset_subset_size=3,
    ...     window_size=None,
    ...     random_state=0,
    ... )
    >>> for i, (train_index, test_index, assets) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train:  index={train_index}")
    ...     print(f"  Test:   index={test_index}")
    ...     print(f"  Assets: columns={assets}")
    Fold 0:
      Train:  index=[0 1]
      Test:   index=[2]
      Assets: columns=[0 1 4]
    Fold 1:
      Train:  index=[1 2]
      Test:   index=[3]
      Assets: columns=[0 1 4]
    Fold 2:
      Train:  index=[0 1]
      Test:   index=[2]
      Assets: columns=[1 3 4]
    Fold 3:
      Train:  index=[1 2]
      Test:   index=[3]
      Assets: columns=[1 3 4]
    >>> print(f"Path ids: {cv.get_path_ids()}")
    Path ids: [0 0 1 1]
    >>>
    >>> # Random contiguous time slice of 4 observations among 10 observations.
    >>> X = np.random.randn(10, 5) # 10 observations and 5 assets.
    >>> cv = MultipleRandomizedCV(
    ...     walk_forward=WalkForward(test_size=1, train_size=2),
    ...     n_subsamples=2,
    ...     asset_subset_size=3,
    ...     window_size=4,
    ...     random_state=0,
    ... )
    >>> for i, (train_index, test_index, assets) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train:  index={train_index}")
    ...     print(f"  Test:   index={test_index}")
    ...     print(f"  Assets: columns={assets}")
    Fold 0:
      Train:  index=[4 5]
      Test:   index=[6]
      Assets: columns=[0 1 4]
    Fold 1:
      Train:  index=[5 6]
      Test:   index=[7]
      Assets: columns=[0 1 4]
    Fold 2:
      Train:  index=[5 6]
      Test:   index=[7]
      Assets: columns=[1 3 4]
    Fold 3:
      Train:  index=[6 7]
      Test:   index=[8]
      Assets: columns=[1 3 4]
    >>>
    >>> # Walk Forward with time-based (calendar) rebalancing.
    >>> # Rebalance every 3 months on the third Friday, and train on the last 12 months.
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> X = X["2021":"2022"]
    >>> cv = MultipleRandomizedCV(
    ...     walk_forward=WalkForward(test_size=3, train_size=12, freq="WOM-3FRI"),
    ...     n_subsamples=2,
    ...     asset_subset_size=3,
    ...     window_size=None,
    ...     random_state=0,
    ... )
    >>> for i, (train_index, test_index, assets) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train:  size={len(train_index)}")
    ...     print(f"  Test:   size={len(test_index)}")
    ...     print(f"  Assets: columns={assets}")
    Fold 0:
      Train:  size=256
      Test:   size=59
      Assets: columns=[ 9 16 17]
    Fold 1:
      Train:  size=253
      Test:   size=61
      Assets: columns=[ 9 16 17]
    Fold 2:
      Train:  size=251
      Test:   size=69
      Assets: columns=[ 9 16 17]
    Fold 3:
      Train:  size=256
      Test:   size=59
      Assets: columns=[ 7 10 14]
    Fold 4:
      Train:  size=253
      Test:   size=61
      Assets: columns=[ 7 10 14]
    Fold 5:
      Train:  size=251
      Test:   size=69
      Assets: columns=[ 7 10 14]
    >>> print(f"Path ids: {cv.get_path_ids()}")
    [0 0 0 1 1 1]
    """

    if TYPE_CHECKING:
        _path_ids: list[int]

    def __init__(
        self,
        walk_forward: WalkForward,
        n_subsamples: int,
        asset_subset_size: int,
        window_size: int | None = None,
        random_state: int | None = None,
    ):
        self.walk_forward = walk_forward
        self.n_subsamples = n_subsamples
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

        assets : ndarray
            The assets indices for that split.
        """
        X, y = sku.indexable(X, y)
        n_observations, n_assets = X.shape
        rng = sku.check_random_state(self.random_state)

        if not isinstance(self.walk_forward, WalkForward):
            raise TypeError("`walk_forward` must be a `WalkForward` instance")

        if self.window_size is not None:
            if not isinstance(self.window_size, int):
                raise ValueError("`window_size` must be an integer")
            if self.window_size < 2 or self.window_size > n_observations:
                raise ValueError(
                    f"When not None, window_size={self.window_size} must "
                    f"satisfy 2 <= window_size <= n_observations={n_observations}."
                )

        if not isinstance(self.asset_subset_size, int):
            raise TypeError("`asset_subset_size` must be an integer")
        if self.asset_subset_size < 1 or self.asset_subset_size > n_assets:
            raise ValueError(
                f"asset_subset_size={self.asset_subset_size} must satisfy "
                f"1 <= asset_subset_size <= n_assets={n_assets}."
            )

        if not isinstance(self.n_subsamples, int):
            raise TypeError("`num_subsamples` must be an integer")

        n_comb = math.comb(n_assets, self.asset_subset_size)
        if self.n_subsamples < 2 or self.n_subsamples > n_comb:
            raise ValueError(
                f"n_subsample={self.n_subsamples} must satisfy 2 <= n_subsample <= "
                f"C({n_assets},{self.asset_subset_size})={n_comb}."
            )

        if n_comb < 1e9:
            asset_indices = sample_unique_subsets(
                n=n_assets,
                k=self.asset_subset_size,
                n_subsets=self.n_subsamples,
                random_state=self.random_state,
            )
        else:
            # Avoid overflow when n_comb is huge (risk of subset duplication is very
            # small and statistically imaterial).
            asset_indices = np.array(
                [
                    rng.choice(n_assets, size=self.asset_subset_size, replace=False)
                    for _ in range(self.n_subsamples)
                ]
            )
            asset_indices.sort(axis=1)

        self._path_ids = []
        for i in range(self.n_subsamples):
            if self.window_size is None:
                start_obs = 0
                X_sample = X
            else:
                start_obs = rng.randint(low=0, high=n_observations - self.window_size)
                obs_indices = np.arange(start_obs, start_obs + self.window_size)
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
            raise ValueError("Before calling `get_path_ids()` you must call `split(X)`")
        return np.array(self._path_ids)
