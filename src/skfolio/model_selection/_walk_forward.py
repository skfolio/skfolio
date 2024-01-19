"""Walk Forward cross-validator"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import sklearn.model_selection as skm
import sklearn.utils as sku


class WalkForward(skm.BaseCrossValidator):
    """Walk Forward cross-validator.

    Provides train/test indices to split time series data samples in a walk forward
    logic.

    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    Compared to `sklearn.model_selection.TimeSeriesSplit`, you control the train/test
    folds by providing a number of training and test samples instead of a number of
    split making it more suitable for portfolio cross-validation.

    Parameters
    ----------
    test_size : int
        Number of observations in each test set.

    train_size : int
        Number of observations in each training set.

    expend_train : bool, default=False
        If this is set to True, each subsequent training set after the first one will
        use all past observations.
        The default is `False`

    reduce_test : bool, default=False
        If this is set to True, the last train/test split will be returned even if the
        test set is partial (if it contains less observations than `test_size`),
        otherwise it will be ignored.
        The default is `False`

    purged_size : int, default=0
        Number of observations to exclude from the end of each train set before the
        test set.
        The default value is `0`

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.model_selection import WalkForward
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
    >>> cv = WalkForward(test_size=1, train_size=2, purged_size=1)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1]
      Test:  index=[3]
    Fold 1:
      Train: index=[1 2]
      Test:  index=[4]
    Fold 2:
      Train: index=[2 3]
      Test:  index=[5]
    >>> cv = WalkForward(test_size=2, train_size=3)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    >>> cv = WalkForward(test_size=2, train_size=3, reduce_test=True)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    Fold 1:
      Train: index=[2 3 4]
      Test:  index=[5]
    >>> cv = WalkForward(test_size=2, train_size=3, expend_train=True, reduce_test=True)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    Fold 1:
      Train: index=[0 1 2 3 4]
      Test:  index=[5]
    """

    def __init__(
        self,
        test_size: int,
        train_size: int,
        expend_train: bool = False,
        reduce_test: bool = False,
        purged_size: int = 0,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.expend_train = expend_train
        self.reduce_test = reduce_test
        self.purged_size = purged_size

    def split(
        self, X: npt.ArrayLike, y=None, groups=None
    ) -> Iterator[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]
        # Make sure we have enough samples for the given split parameters
        if self.train_size + self.purged_size >= n_samples:
            raise ValueError(
                "The sum of `train_size` with `purged_size` "
                f"({self.train_size + self.purged_size}) cannot be greater than the"
                f" number of samples ({n_samples})."
            )

        indices = np.arange(n_samples)

        test_start = self.train_size + self.purged_size
        while True:
            if test_start >= n_samples:
                return
            test_end = test_start + self.test_size
            train_end = test_start - self.purged_size
            if self.expend_train:
                train_start = 0
            else:
                train_start = train_end - self.train_size

            if test_end > n_samples:
                if not self.reduce_test:
                    return
                yield (
                    indices[train_start:train_end],
                    indices[test_start:],
                )
            else:
                yield (
                    indices[train_start:train_end],
                    indices[test_start:test_end],
                )
            test_start = test_end

    def get_n_splits(self, X: npt.ArrayLike, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
         X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_folds : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]
        n = n_samples - self.train_size - self.purged_size

        if self.reduce_test and n % self.test_size != 0:
            return n // self.test_size + 1
        return n // self.test_size
