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
import pandas as pd
import sklearn.model_selection as sks
import sklearn.utils as sku


class WalkForward(sks.BaseCrossValidator):
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

    train_size : int | offset
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
        train_size: int | pd.tseries.offsets.BaseOffset,
        period: str | None = None,
        expend_train: bool = False,
        reduce_test: bool = False,
        purged_size: int = 0,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.period = period
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

        if not isinstance(self.test_size, int):
            raise ValueError("test_size` must be an integer")

        if self.period is None:
            if not isinstance(self.train_size, int):
                raise ValueError(
                    "When `period` is None, `train_size` must be an integer"
                )
            return _split_without_period(
                n_samples=n_samples,
                train_size=self.train_size,
                test_size=self.test_size,
                purged_size=self.purged_size,
                expend_train=self.expend_train,
                reduce_test=self.reduce_test,
            )

        if not hasattr(X, "index") or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must be a DataFrame with an index of type DatetimeIndex"
            )
        if isinstance(self.train_size, int):
            return _split_from_period_without_train_offset(
                n_samples=n_samples,
                train_size=self.train_size,
                test_size=self.test_size,
                period=self.period,
                purged_size=self.purged_size,
                expend_train=self.expend_train,
                reduce_test=self.reduce_test,
                ts_index=X.index,
            )
        return _split_from_period_with_train_offset(
            n_samples=n_samples,
            train_size=self.train_size,
            test_size=self.test_size,
            period=self.period,
            purged_size=self.purged_size,
            expend_train=self.expend_train,
            reduce_test=self.reduce_test,
            ts_index=X.index,
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
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


def _split_without_period(
    n_samples: int,
    train_size: int,
    test_size: int,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
) -> Iterator[np.ndarray, np.ndarray]:
    if train_size + purged_size >= n_samples:
        raise ValueError(
            "The sum of `train_size` with `purged_size` "
            f"({train_size + purged_size}) cannot be greater than the"
            f" number of samples ({n_samples})."
        )

    indices = np.arange(n_samples)

    test_start = train_size + purged_size
    while True:
        if test_start >= n_samples:
            return
        test_end = test_start + test_size
        train_end = test_start - purged_size
        if expend_train:
            train_start = 0
        else:
            train_start = train_end - train_size

        if test_end > n_samples:
            if not reduce_test:
                return
            test_indices = indices[test_start:]
        else:
            test_indices = indices[test_start:test_end]
        train_indices = indices[train_start:train_end]
        yield train_indices, test_indices

        test_start = test_end


def _split_from_period_without_train_offset(
    n_samples: int,
    train_size: int,
    test_size: int,
    period: str,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
    ts_index,
) -> Iterator[np.ndarray, np.ndarray]:
    date_range = pd.date_range(start=ts_index[0], end=ts_index[-1], freq=period)
    idx = ts_index.get_indexer(date_range, method="ffill")
    n = len(idx)
    i = 0
    while True:
        if i + train_size >= n:
            return

        if i + train_size + test_size >= n:
            if not reduce_test:
                return
            test_indices = np.arange(idx[i + train_size], n_samples)

        else:
            test_indices = np.arange(
                idx[i + train_size], idx[i + train_size + test_size]
            )
        if expend_train:
            train_start = 0
        else:
            train_start = idx[i]
        train_indices = np.arange(train_start, idx[i + train_size] - purged_size)
        yield train_indices, test_indices

        i += test_size


def _split_from_period_with_train_offset(
    n_samples: int,
    train_size: pd.tseries.offsets.BaseOffset,
    test_size: int,
    period: str,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
    ts_index,
) -> Iterator[np.ndarray, np.ndarray]:
    date_range = pd.date_range(start=ts_index[0], end=ts_index[-1], freq=period)
    idx = ts_index.get_indexer(date_range, method="ffill")
    train_idx = ts_index.get_indexer(date_range - train_size, method="ffill")

    n = len(idx)

    i = np.argmax(train_idx > -1)
    while True:
        if i >= n:
            return

        if i + test_size >= n:
            if not reduce_test:
                return
            test_indices = np.arange(idx[i], n_samples)
        else:
            test_indices = np.arange(idx[i], idx[i + test_size] - purged_size)

        if expend_train:
            train_start = 0
        else:
            train_start = train_idx[i]
        train_indices = np.arange(train_start, idx[i])
        yield train_indices, test_indices

        i += test_size
