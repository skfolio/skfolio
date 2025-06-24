"""Multiple Randomized CV."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credit: Daniel Palomar
# SPDX-License-Identifier: BSD-3-Clause

import random
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import sklearn.utils as sku

from skfolio.model_selection._walk_forward import WalkForward
from skfolio.utils.tools import safe_split


class MultipleRandomizedCV:
    """Multiple Randomized Cross-Validation.

    TODO

    Parameters
    ----------
    walk_forward : WalkForward
        Walk Forward cv instance applied to each sub-sample.

    n_sample_observations : int
        Number of **contigous** observations to samples from the entire dataset.
        Must be less or equal to the total number of observations.
        If None, all observations are kept in each sun-sample.

    n_sample_assets : int
        Number of assets to samples from the entire dataset.
        Must be less or equal to the total number of assets.
        If None, all assets are kept in each sun-sample.

    n_subsamples : int, default=10
        Number of sub-sample datasets of shape (n_sample_observations, n_sample_assets)
        on which Walk Forward is ran. The default is 10.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Attributes
    ----------
    index_train_test_ : ndarray of shape (n_observations, n_splits)

    Examples
    --------
    TODO

    References
    ----------
    .. [1]  "Portfolio Optimization, Theory and Application",
        Daniel P. Palomar (2025)
    """

    index_train_test_: np.ndarray

    def __init__(
        self,
        walk_forward: WalkForward,
        n_sample_observations: int,
        n_sample_assets: int,
        n_subsamples: int = 10,
        random_state: int | None = None,
    ):
        self.walk_forward = walk_forward
        self.n_sample_observations = n_sample_observations
        self.n_sample_assets = n_sample_assets
        self.n_subsamples = n_subsamples
        self.random_state = random_state

    def split(
        self, X: npt.ArrayLike, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, list[np.ndarray]]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,), optional
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
            raise ValueError("walk_forward` must be a `WalkForward` instance")
        if not isinstance(self.n_sample_observations, int):
            raise ValueError("n_sample_observations` must be an integer")
        if not isinstance(self.n_sample_assets, int):
            raise ValueError("n_sample_assets` must be an integer")
        if not isinstance(self.n_subsamples, int):
            raise ValueError("n_subsamples` must be an integer")

        if self.n_sample_observations > n_observations:
            raise ValueError

        if self.n_sample_assets > n_assets:
            raise ValueError

        # TODO
        # math.comb(n_assets, self.n_sample_assets)
        # n_observations-self.n_sample_observations

        asset_ids = range(n_assets)

        for _ in range(self.n_subsamples):
            asset_indices = rng.choice(
                asset_ids, size=self.n_sample_assets, replace=False
            )
            asset_indices.sort()
            rng.choice(asset_ids, size=self.n_sample_assets, replace=False)
            start_obs = random.randint(0, n_observations - self.n_sample_observations)
            obs_indices = np.arange(start_obs, start_obs + self.n_sample_observations)
            X_sample, _ = safe_split(
                X,
                indices=obs_indices,
                axis=0,
            )
            for train_indices, test_indices in self.walk_forward.split(X_sample):
                yield train_indices, test_indices, asset_indices
